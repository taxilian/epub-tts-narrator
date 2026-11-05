import unittest

from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.renderer.performance_runtime import row_performance


class InlinePlanOrderTests(unittest.TestCase):
    def test_external_control_before_inline_control_merges_without_a_false_overlap(self):
        text = "First second third"
        row = {"id": "s1", "text": text, "start": 0, "end": len(text),
               "delivery": {"direction": "", "evidence": []},
               "performance_controls": {"version": 1, "text_sha256": text_sha256(text), "controls": [
                   {"kind": "delivery", "tag": "whisper", "direction": "", "start": 13,
                    "end": 18, "text": "third"}]}}
        external = {"version": 1, "text_sha256": text_sha256(text), "controls": [
            {"kind": "delivery", "tag": "soft", "direction": "", "start": 0,
             "end": 5, "text": "First"}]}

        merged = row_performance(row, {"s1": external}, normalization=False)

        self.assertEqual([(control["start"], control["end"], control["tag"])
                          for control in merged["controls"]], [(0, 5, "soft"), (13, 18, "whisper")])
