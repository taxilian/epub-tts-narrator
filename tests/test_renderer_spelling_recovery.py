"""A chosen name misread with its phoneme tags is sent as written, and back again."""

import unittest

from epub_tts_narrator.renderer.spelling_recovery import (
    marker, plain_next, plain_spelling_request, validate_marker)
from test_renderer_name_shape import performance

TEXT = "Marendra nodded."


def request(controls=None):
    record = performance(TEXT, "Marendra", "M AH0 R EH1 N D R AH0")
    if controls is not None:
        record["controls"] = controls
    return {"kind": "speech", "chunk": {"spoken_text": TEXT, "expressive_performance": record}}


def take(reasons, *, plain=False, audio=True):
    metadata = {"spelling_recovery": {}} if plain else {}
    return {"report": {"passed": not reasons, "reasons": list(reasons)}, "engine_metadata": metadata,
            "audio_file": "take.wav" if audio else None}


class PlainSpellingTests(unittest.TestCase):
    def test_only_chosen_phoneme_tags_are_left_out(self):
        style = {"kind": "delivery", "start": 9, "end": 16, "text": "nodded."}
        original = request(request()["chunk"]["expressive_performance"]["controls"] + [style])
        plain = plain_spelling_request(original)
        self.assertEqual(plain["chunk"]["expressive_performance"]["controls"], [style])
        self.assertEqual(plain["chunk"]["spoken_text"], TEXT)
        self.assertIsNone(plain_spelling_request(request([style])))  # nothing to leave out

    def test_a_take_record_must_name_its_own_request(self):
        original = request()
        self.assertEqual(validate_marker(original, marker(original)), plain_spelling_request(original))
        other = request()
        other["chunk"]["spoken_text"] = "Marendra smiled."
        with self.assertRaisesRegex(ValueError, "differs from its original request"):
            validate_marker(other, marker(original))

    def test_the_passages_own_misreads_choose_tags_or_spelling(self):
        misread = ["pronunciation_mismatch"]
        self.assertFalse(plain_next([take(misread)]))
        self.assertTrue(plain_next([take(misread), take(misread)]))
        # Plain spelling misread as often: tags come back, and so on.
        self.assertFalse(plain_next([take(misread)] * 2 + [take(misread, plain=True)] * 2))
        self.assertTrue(plain_next([take(misread)] * 3 + [take(misread, plain=True)] * 2))
        # Only misread names count, and only takes with audio.
        self.assertFalse(plain_next([take(["text_mismatch"])] * 3))
        self.assertFalse(plain_next([take(misread, audio=False)] * 3))


if __name__ == "__main__":
    unittest.main()
