"""Normalized performances retain source links across rendering and auditions."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.bundle import make_plan
from epub_tts_narrator.renderer.chapter_protocol import validate_spec
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.samples import select_book_sample
from epub_tts_narrator.spoken_text import build_spoken, effective_delivery, spoken_source_ranges
from test_renderer_samples import bundle, row


def normalize(passage, original, replacement):
    start = passage["text"].index(original)
    passage["speech_normalization"] = {"version": 1, "edits": [{
        "start": start, "end": start + len(original), "original": original,
        "replacement": replacement, "direction": "Draw out the vowel naturally.",
        "reason": "Repeated spelling conveys a sustained vowel.",
    }]}
    passage["spoken_text"] = passage["text"].replace(original, replacement, 1)
    return passage


class NormalizedSpeechRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def plan(self, passage, maximum=1000):
        data = bundle([passage])
        data.update(directory=str(Path(self.temp.name) / "book"), bundle_hash="b" * 64)
        data["manifest"]["source_fingerprint"] = "a" * 64
        data["chapters"][0]["entry"]["chapter_id"] = "ch1"
        data["reviews"] = []
        config = deepcopy(DEFAULTS)
        config["render"]["max_chunk_chars"] = maximum
        return make_plan(data, config)

    def test_length_changing_normalization_reaches_renderer_with_original_source_range(self):
        passage = normalize(row("Nooooooo!", start=17), "Nooooooo", "No")
        original = deepcopy(passage)
        plan = self.plan(passage)
        chunk = plan["chapters"][0]["chunks"][0]
        self.assertEqual(chunk["spoken_text"], "No!")
        self.assertEqual(chunk["source_ranges"], [{"start": 17, "end": 26}])
        self.assertEqual(chunk["segment"], original)
        self.assertEqual(passage, original)

    def test_source_audition_uses_normal_spelling_and_retains_source(self):
        text = ("Thisss is the road that brought us here. There is still time to find our friends "
                "and bring them safely home before nightfall. We know which path will lead us there.")
        passage = normalize(row(text, start=12), "Thisss", "This")
        selected = select_book_sample(bundle([passage]), "v1", canonical_id=lambda value: value)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["reference_text"], passage["spoken_text"])
        self.assertEqual(selected["source"]["source_ranges"],
                         [{"start": 12, "end": 12 + len(text)}])

    def test_extreme_normalized_delivery_is_not_chosen_for_reusable_voice_sample(self):
        text = ("Nooooooo, I remember the road that brought us here. There is still time to find our friends "
                "and bring them safely home before nightfall. We know which path will lead us there.")
        passage = normalize(row(text), "Nooooooo", "No")
        passage["speech_normalization"]["edits"][0]["direction"] = "Scream the first word."
        selected = select_book_sample(bundle([passage]), "v1", canonical_id=lambda value: value)
        self.assertIsNone(selected)

    def test_effective_delivery_keeps_existing_style_and_unique_approved_styles(self):
        passage = normalize(row("Nooooooo! Thisss cannot happen."), "Nooooooo", "No")
        passage["delivery"]["direction"] = "Quietly"
        edit = passage["speech_normalization"]["edits"][0]
        second = dict(edit, start=10, end=16, original="Thisss", replacement="This")
        passage["speech_normalization"]["edits"].append(second)
        original = deepcopy(passage)
        self.assertEqual(effective_delivery(passage), "Quietly; Draw out the vowel naturally.")
        self.assertEqual(passage, original)

    def test_overlong_combined_delivery_fails_instead_of_truncating_style(self):
        passage = normalize(row("Nooooooo!"), "Nooooooo", "No")
        passage["delivery"]["direction"] = "softly " * 175
        with self.assertRaisesRegex(ValueError, "Combined speech delivery exceeds"):
            effective_delivery(passage)

    def test_chunk_cut_inside_expanded_spelling_retains_complete_original_word(self):
        passage = normalize(row("Denno, Vern.", start=20), "Denno", "Don't know")
        chunks = self.plan(passage, maximum=3)["chapters"][0]["chunks"]
        self.assertEqual("".join(chunk["spoken_text"] for chunk in chunks), "Don't know, Vern.")
        for chunk in chunks:
            if chunk["spoken_start"] < len("Don't know"):
                self.assertEqual(chunk["source_ranges"][0]["start"], 20)
                self.assertGreaterEqual(chunk["source_ranges"][0]["end"], 25)
            self.assertLessEqual(len(chunk["spoken_text"]), 3)

    def test_brackets_and_spelling_replacements_compose_without_losing_source_links(self):
        passage = normalize(row("[Thisss [iss] odd]", start=30), "Thisss", "This")
        passage["performance"] = {"kind": "bracketed"}
        passage["spoken_text"] = "This iss odd"
        spoken, _ = build_spoken(passage)
        self.assertEqual(spoken, "This iss odd")
        self.assertEqual(spoken_source_ranges(passage, 0, len(spoken)), [
            {"start": 31, "end": 38}, {"start": 39, "end": 42},
            {"start": 43, "end": 47},
        ])
        # A subword highlight must not invent a precise character mapping for a
        # spelling that was replaced; it points at the entire original word.
        self.assertEqual(spoken_source_ranges(passage, 1, 3), [{"start": 31, "end": 37}])

    def test_unchanged_text_around_replacement_still_has_precise_unicode_offsets(self):
        passage = normalize(row("Élise: nooooo, 水!", start=9), "nooooo", "no")
        spoken, _ = build_spoken(passage)
        start = spoken.index("水")
        source_start = 9 + passage["text"].index("水")
        self.assertEqual(spoken_source_ranges(passage, start, start + 1),
                         [{"start": source_start, "end": source_start + 1}])

    def test_arbitrary_spoken_rewrite_cannot_bypass_saved_edits(self):
        passage = normalize(row("Nooooooo!", start=17), "Nooooooo", "No")
        passage["spoken_text"] = "Yes!"
        with self.assertRaises(ValueError):
            self.plan(passage)

    def test_invalid_edit_records_cannot_relabel_the_source_or_inject_controls(self):
        original = normalize(row("Nooooooo!", start=17), "Nooooooo", "No")
        changes = (
            {"start": True}, {"end": 1000}, {"original": "different"},
            {"replacement": ""}, {"replacement": "[speaker:1] No"},
            {"replacement": "<whisper>No</whisper>"}, {"replacement": "N\x00o"},
            {"direction": "say this\nthen that"}, {"reason": ""},
        )
        for change in changes:
            with self.subTest(change=change):
                passage = deepcopy(original)
                passage["speech_normalization"]["edits"][0].update(change)
                with self.assertRaises(ValueError):
                    self.plan(passage)

    def test_overlapping_edits_are_rejected_before_speech_is_planned(self):
        passage = normalize(row("Nooooooo!", start=17), "Nooooooo", "No")
        passage["speech_normalization"]["edits"].append({
            "start": 2, "end": 7, "original": "ooooo", "replacement": "oh",
            "direction": "A drawn out vowel.", "reason": "Sustained sound.",
        })
        with self.assertRaises(ValueError):
            self.plan(passage)

    def test_remote_worker_accepts_approved_mapping_and_rejects_forged_source_link(self):
        passage = normalize(row("Nooooooo!", start=17), "Nooooooo", "No")
        plan = self.plan(passage)
        spec = {"version": 1, "book": plan["book"], "source_fingerprint": "a" * 64,
                "renderer_fingerprint": "b" * 64, "chapter": plan["chapters"][0],
                "total_chapters": 1, "voice_bindings": {"v1": "c" * 64},
                "render": {"chapter_format": "m4a"}}
        self.assertEqual(validate_spec(spec), spec)
        spec["chapter"]["chunks"][0]["source_ranges"] = [{"start": 17, "end": 19}]
        with self.assertRaises(ValueError):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
