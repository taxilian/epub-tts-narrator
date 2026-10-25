from copy import deepcopy
import unittest

from epub_tts_narrator.performance_plan import (
    combine_performances, slice_performance, text_sha256, validate_performance,
)


def record(text, *controls):
    return {"version": 1, "text_sha256": text_sha256(text), "controls": list(controls)}


def delivery(text, start=0, end=None, tag="whisper", **extra):
    end = len(text) if end is None else end
    return {"kind": "delivery", "start": start, "end": end, "text": text[start:end], "tag": tag, **extra}


def pronunciation(text, start, end, phonemes="JH AE1 S"):
    return {"kind": "pronunciation", "start": start, "end": end, "text": text[start:end],
            "alphabet": "cmu_arpabet", "phonemes": phonemes}


class PerformancePlanTests(unittest.TestCase):
    def test_provenance_does_not_change_effective_identity_or_mutate_record(self):
        text = "Jas spoke."
        plan = record(text, delivery(text, reason="Source says whispered", evidence=[{"block_id": "b1", "quote": "he whispered"}]))
        plan["reason"] = "Reviewed"
        original = deepcopy(plan)
        result = validate_performance(plan, text)
        self.assertEqual(plan, original)
        self.assertNotIn("reason", result)
        self.assertNotIn("evidence", result["controls"][0])
        self.assertEqual(result, validate_performance(record(text, delivery(text)), text))

    def test_binding_rejects_changed_text_and_boolean_version(self):
        for text, version in (("Changed.", 1), ("Original.", True)):
            plan = record("Original.")
            plan["version"] = version
            with self.subTest(text=text, version=version), self.assertRaises(ValueError):
                validate_performance(plan, text)

    def test_spans_use_unicode_code_points_and_exact_anchors(self):
        text = "Éowyn 🐉 said go."
        result = validate_performance(record(text, delivery(text, 6, 7, tag="emphasis")), text)
        self.assertEqual(result["controls"][0]["text"], "🐉")
        invalid = record(text, delivery(text, 6, 7))
        invalid["controls"][0]["text"] = "said"
        with self.assertRaises(ValueError):
            validate_performance(invalid, text)

    def test_unknown_fields_and_markup_directions_fail_closed(self):
        text = "Do come."
        for control in (delivery(text, ignored=True), delivery(text, direction="[speaker:1]"),
                        delivery(text, direction="soft\nvoice"), delivery(text, tag="shout"),
                        delivery(text, tag="style")):
            with self.subTest(control=control), self.assertRaises(ValueError):
                validate_performance(record(text, control), text)
        plan = record(text)
        plan["unknown"] = True
        with self.assertRaises(ValueError):
            validate_performance(plan, text)

    def test_malformed_kinds_fail_with_validation_errors(self):
        for kind in (None, [], {}, True, 1):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_performance(record("Go.", {"kind": kind}), "Go.")

    def test_phonemes_allow_standard_cmu_stress_and_reject_other_alphabets(self):
        text = "Jas spoke."
        for phones in ("JH AE1 S", "JH AE S", "JH AE2 S"):
            with self.subTest(phones=phones):
                validate_performance(record(text, pronunciation(text, 0, 3, phones)), text)
        for phones in ("JH AE3 S", "JH1 AE1 S", "dʒ æ z", "JH  AE1 S", "JH AE1\nS", "<whisper>"):
            with self.subTest(phones=phones), self.assertRaises(ValueError):
                validate_performance(record(text, pronunciation(text, 0, 3, phones)), text)
        with self.assertRaises(ValueError):
            validate_performance(record(text, pronunciation(text, 0, 9)), text)
        with self.assertRaisesRegex(ValueError, "part of a word"):
            validate_performance(record(text, pronunciation(text, 0, 2)), text)

    def test_pronunciation_may_be_nested_in_delivery_but_not_cross_its_boundary(self):
        text = "Jas spoke."
        validate_performance(record(text, delivery(text), pronunciation(text, 0, 3)), text)
        with self.assertRaisesRegex(ValueError, "cross"):
            validate_performance(record(text, delivery(text, 2, 6), pronunciation(text, 0, 3)), text)

    def test_overlapping_delivery_ranges_are_rejected_not_arbitrarily_prioritized(self):
        text = "Jas spoke."
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_performance(record(text, delivery(text), delivery(text, 0, 3, tag="emphasis")), text)

    def test_event_inside_pronunciation_rejected_but_at_boundary_allowed(self):
        text = "Jas spoke."
        for offset, valid in ((0, True), (2, False), (3, True)):
            plan = record(text, pronunciation(text, 0, 3), {"kind": "pause", "offset": offset, "duration_ms": 250})
            with self.subTest(offset=offset):
                if valid:
                    validate_performance(plan, text)
                else:
                    with self.assertRaises(ValueError):
                        validate_performance(plan, text)

    def test_event_values_are_bounded_and_duplicate_events_fail(self):
        text = "Go."
        for value in (True, 0, 99, 5001, 200.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_performance(record(text, {"kind": "pause", "offset": 0, "duration_ms": value}), text)
        event = {"kind": "vocalization", "offset": 0, "sound": "sigh"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_performance(record(text, event, event), text)

    def test_slicing_clips_delivery_and_assigns_each_boundary_event_once(self):
        text = "one two"
        plan = record(text, delivery(text), {"kind": "pause", "offset": 4, "duration_ms": 400},
                      {"kind": "vocalization", "offset": 7, "sound": "sigh"})
        first = slice_performance(plan, text, 0, 4)
        second = slice_performance(plan, text, 4, 7)
        self.assertEqual(first["controls"], [{**delivery("one "), "direction": ""}])
        self.assertEqual([c["offset"] for c in second["controls"] if "offset" in c], [0, 3])
        self.assertEqual(second["text_sha256"], text_sha256("two"))

    def test_slicing_cannot_fragment_a_pronunciation(self):
        text = "Jas spoke."
        plan = record(text, pronunciation(text, 0, 3))
        with self.assertRaisesRegex(ValueError, "cut through"):
            slice_performance(plan, text, 1, len(text))
        self.assertEqual(slice_performance(plan, text, 0, 3)["controls"][0]["text"], "Jas")

    def test_combine_rebases_spans_and_events_without_rewriting_words(self):
        first, second = "Stay.", "Go."
        result = combine_performances([(first, record(first, delivery(first))),
                                      (second, record(second, {"kind": "pause", "offset": 0, "duration_ms": 300}))])
        self.assertEqual(result["text_sha256"], text_sha256("Stay. Go."))
        self.assertEqual(result["controls"][1]["offset"], 6)
