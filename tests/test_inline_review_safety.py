from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.attribution_performance import review_inline_performance
from epub_tts_narrator.inline_performance import project_segments
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.pipeline import CachedRequests, apply_edits, fingerprint
from epub_tts_narrator.spoken_text import build_spoken
from test_attribution_performance import ReviewClient, document, segment
from test_contracts import valid_document


class InlineReviewSafetyTests(unittest.TestCase):
    def test_partial_token_edits_cannot_change_protected_tokens_or_keep_dependent_controls(self):
        cases = [("Mara said.", "a", "M", "ra", "e"),
                 ("never there.", "v", "ne", "er", "b"),
                 ("one spoon.", "o", "", "ne", "a")]
        for text, original, left, right, replacement in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temporary:
                token = text.split()[0]
                row = segment("b1", text, proposals={
                    "edits": [{"original": original, "left": left, "right": right,
                               "replacement": replacement, "direction": "colloquial accent",
                               "reason": "A proposed partial spelling change."}],
                    "controls": [{"kind": "delivery", "tag": "whisper", "original": token,
                                  "reason": "This cue depends on the proposed word."}]})
                client = ReviewClient()
                result = review_inline_performance(document([row], block_text=text,
                    characters=[{"name": "Mara", "aliases": []}]),
                    CachedRequests(client, Path(temporary), repair_attempts=0))

                self.assertEqual(client.calls, [])
                self.assertNotIn("speech_normalization", result["segments"][0])
                self.assertNotIn("performance_controls", result["segments"][0])
                self.assertEqual(build_spoken(result["segments"][0])[0], text)

    def test_a_discarded_edit_does_not_remove_an_unrelated_control_in_another_block(self):
        first = segment("b1", "never there.", proposals={"edits": [{
            "original": "never", "replacement": "ever", "direction": "plainly", "reason": "Invalid change."}]})
        second = segment("b2", "Keep quiet.", proposals={"controls": [{
            "kind": "delivery", "tag": "whisper", "original": "Keep", "reason": "A quiet aside."}]})
        source = document([first, second], block_text=first["text"])
        source["source"]["blocks"].append({"id": "b2", "chapter_id": "ch1", "text": second["text"], "source": {}})
        with tempfile.TemporaryDirectory() as temporary:
            result = review_inline_performance(source, CachedRequests(ReviewClient(), Path(temporary), repair_attempts=0))

        self.assertNotIn("speech_normalization", result["segments"][0])
        self.assertEqual(result["segments"][1]["performance_controls"]["controls"][0]["text"], "Keep")

    def saved_document(self):
        saved = valid_document()
        saved["cast"]["roles"][0]["id"] = "narrator_main"
        for row in saved["segments"]:
            if row["role_id"] == "narrator":
                row["role_id"] = "narrator_main"
        text = "Wull now."
        saved["source"]["blocks"][1]["text"] = text
        row = saved["segments"][-1]
        row.update(text=text, start=0, end=len(text))
        row["speech_normalization"] = {"version": 1, "edits": [{"start": 0, "end": 4,
            "original": "Wull", "replacement": "Well", "direction": "colloquial accent", "reason": "Saved edit."}]}
        row["spoken_text"] = "Well now."
        row["performance_controls"] = {"version": 1, "text_sha256": text_sha256(row["spoken_text"]),
            "controls": [{"kind": "delivery", "tag": "whisper", "direction": "",
                          "start": 0, "end": 4, "text": "Well"}]}
        return saved

    def test_a_cut_edit_drops_dependent_styles_instead_of_applying_them_to_source_fragments(self):
        old = self.saved_document()["segments"][-1]
        new = []
        for start, end in ((0, 2), (2, len(old["text"]))):
            row = deepcopy(old)
            row.update(id=f"b2:{start}:{end}", text=old["text"][start:end], start=start, end=end)
            for key in ("speech_normalization", "spoken_text", "performance_controls"):
                row.pop(key, None)
            new.append(row)
        drops = []

        projected = project_segments([old], new, drops=drops)

        self.assertTrue(all("speech_normalization" not in row and "performance_controls" not in row
                            for row in projected))
        self.assertEqual("".join(row["text"] for row in projected), old["text"])
        self.assertEqual({item["kind"] for item in drops}, {"edit", "delivery"})

    def test_manual_block_corrections_preserve_approved_metadata_at_same_or_new_boundaries(self):
        for pieces in (("Wull now.",), ("Wull ", "now.")):
            with self.subTest(pieces=pieces):
                saved = self.saved_document()
                edits = {"source_fingerprint": fingerprint(saved["source"]),
                         "cast_fingerprint": fingerprint(saved["cast"]),
                         "blocks": [{"block_id": "b2", "segments": [
                             {"text": text, "kind": "narration", "role_id": "narrator_main"} for text in pieces]}]}

                updated = apply_edits(saved, edits)

                rows = [row for row in updated["segments"] if row["block_id"] == "b2"]
                self.assertEqual(rows[0]["speech_normalization"], saved["segments"][-1]["speech_normalization"])
                self.assertEqual(rows[0]["performance_controls"]["controls"][0]["text"], "Well")
                self.assertEqual("".join(build_spoken(row)[0] for row in rows), "Well now.")
                self.assertEqual(updated["source"], saved["source"])
