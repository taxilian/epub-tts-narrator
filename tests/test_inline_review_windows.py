from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.attribution_performance import review_inline_performance
from epub_tts_narrator.pipeline import CachedRequests
from test_attribution_performance import ReviewClient, document, segment


class InlineReviewWindowTests(unittest.TestCase):
    def review(self, row):
        client = ReviewClient()
        with tempfile.TemporaryDirectory() as temporary:
            requests = CachedRequests(client, Path(temporary), repair_attempts=0)
            result = review_inline_performance(document([row], block_text=row["text"]), requests)
        return result, client.calls

    def test_second_review_batch_of_the_same_segment_keeps_its_source_context(self):
        words = [f"word{index:02d}" for index in range(26)]
        text = " ".join(words)
        row = segment("b1", text, proposals={"controls": [
            {"kind": "delivery", "tag": "emphasis", "original": word, "reason": "The word is emphasized."}
            for word in words]})

        _, calls = self.review(row)

        self.assertEqual([len(call["proposals"]) for call in calls], [24, 2])
        for call in calls:
            visible = " ".join(item["text"] for item in call["segments"])
            for proposal in call["proposals"]:
                self.assertIn(proposal["locator"]["original"], visible)

    def test_distant_edits_in_a_long_segment_are_never_reviewed_against_a_truncated_away_span(self):
        text = "Wull " + "plain " * 2000 + "rekkon."
        row = segment("b1", text, proposals={"edits": [
            {"original": original, "replacement": replacement, "direction": "colloquial accent",
             "reason": "The source uses a dialect spelling."}
            for original, replacement in (("Wull", "Well"), ("rekkon", "reckon"))]})

        _, calls = self.review(row)

        self.assertEqual(len(calls), 2)
        for call in calls:
            visible = " ".join(item["text"] for item in call["segments"])
            self.assertLessEqual(sum(len(item["text"]) for item in call["segments"] + call["context"]), 6000)
            for proposal in call["proposals"]:
                self.assertIn(proposal["locator"]["original"], visible)
