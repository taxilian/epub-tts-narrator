from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.attribution_performance import review_inline_performance
from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.renderer.performance_runtime import row_performance
from epub_tts_narrator.renderer.pronunciation_runtime import merge_row_pronunciations
from test_attribution_performance import ReviewClient, document, segment


class InlineEditDeliveryTests(unittest.TestCase):
    def test_edit_only_delivery_stays_on_the_edited_word_without_an_explicit_style_control(self):
        text = "Noooo, stop."
        row = segment("b1", text, proposals={"edits": [{
            "original": "Noooo", "replacement": "No", "direction": "sustain the vowel",
            "reason": "The source stretches this word only.",
        }]})
        with tempfile.TemporaryDirectory() as temporary:
            requests = CachedRequests(ReviewClient(), Path(temporary), repair_attempts=0)
            result = review_inline_performance(document([row], block_text=text), requests)

        published = result["segments"][0]
        record = row_performance(published, None, normalization=False)
        self.assertIsNotNone(record, "new inline edit directions need structured source-bound binding")
        record = merge_row_pronunciations(published, record, {})
        self.assertEqual([(control["text"], control["start"], control["end"])
                          for control in record["controls"] if control["kind"] == "delivery"], [("No", 0, 2)])
        self.assertEqual(published["text"], text)
        self.assertEqual(published["spoken_text"], "No, stop.")
