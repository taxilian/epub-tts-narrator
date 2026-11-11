"""Another book's chosen voices are reused for the same characters, matched by name."""

from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import voice_transfer
from epub_tts_narrator.renderer.voice_transfer import match_voices


def book(directory, characters, narrator_voice="vn", derived=()):
    """A bundle-shaped cast: characters are (name, aliases, voice id)."""
    voices = [{"id": narrator_voice, "description": "Narration."}]
    voices += [{"id": voice, "description": name} for name, _, voice in characters]
    voices += [{"id": voice, "description": "Older.", "derived_from": base} for voice, base in derived]
    return {"directory": directory, "source": {"book": {"id": directory}},
            "cast": {"characters": [{"id": f"c-{voice}", "name": name, "aliases": list(aliases), "voice_id": voice}
                                    for name, aliases, voice in characters],
                     "voices": voices,
                     "roles": [{"id": "narrator_main", "kind": "narration", "voice_id": narrator_voice}]}}


class MatchTests(unittest.TestCase):
    def test_characters_match_by_name_or_one_shared_alias_never_by_id(self):
        old = book("old", [("Belgarath", ["Mister Wolf", "Old Wolf"], "o1"), ("Garion", [], "o2"),
                           ("Silk", ["Prince Kheldar"], "o3"), ("Brill", [], "o4"), ("Brill", [], "o5")],
                   narrator_voice="on")
        new = book("new", [("Belgarath", [], "n1"), ("Kheldar", ["Silk"], "n3"), ("Old Wolf", [], "n6"),
                           ("Brill", [], "n4"), ("Zedar", [], "n7"), ("Garion", [], "n2")],
                   derived=[("n2-older", "n2")])
        matches, unmatched = match_voices(new, old)
        self.assertEqual(sorted((ours, theirs) for ours, theirs, _ in matches),
                         [("n1", "o1"), ("n2", "o2"), ("n3", "o3"), ("n6", "o1"), ("vn", "on")])
        self.assertEqual({label: reason for _, label, reason in unmatched},
                         {"Brill": "several voices share its names", "Zedar": "no character of that name"})


class FakeStore:
    def __init__(self, saved):
        self.saved = saved

    def load(self, voice_id):
        return self.saved.get(voice_id)


class ImportTests(unittest.TestCase):
    def test_matched_voices_import_their_reference_and_traits_and_existing_choices_stay(self):
        old = book("/books/old", [("Mara", [], "o1"), ("Tom", [], "o2"), ("Ned", [], "o3")], narrator_voice="on")
        new = book("/books/new", [("Mara", [], "n1"), ("Tom", [], "n2"), ("Ned", [], "n3")])
        stores = {"/books/old": FakeStore({"o1": {"reference_audio": "/v/mara-reference.wav", "reference_text": "Hi."},
                                           "o2": {"reference_audio": "/v/tom-reference.wav", "reference_text": "Yo."},
                                           "o3": {"reference_audio": "/v/same.wav", "reference_text": "Ok.",
                                                  "reference_sha256": "abc"},
                                           "on": {"reference_audio": "/v/narrator-reference.wav",
                                                  "reference_text": "Once."}}),
                  "/books/new": FakeStore({"n2": {"reference_audio": "/v/new-tom.wav", "reference_text": "Hey."},
                                           "vn": {"reference_audio": "/v/inherited.wav", "reference_text": "So.",
                                                  "inherited": True},
                                           "n3": {"reference_audio": "/v/same.wav", "reference_sha256": "abc",
                                                  "inherited": True}})}
        imported, traits, lines = [], [], []
        with (patch.object(voice_transfer, "load_bundle", return_value=old),
              patch("epub_tts_narrator.renderer.book_context.resolve_book_config", side_effect=lambda b, c: c),
              patch("epub_tts_narrator.renderer.book_context.voice_store",
                    side_effect=lambda b, c: stores[b["directory"]]),
              patch("epub_tts_narrator.renderer.speech_traits.speech_trait_overrides",
                    side_effect=lambda b, c: {"o1": "Light Irish accent."} if b is old else {}),
              patch("epub_tts_narrator.renderer.speech_traits.save_speech_traits",
                    side_effect=lambda b, c, voice, value: traits.append((voice, value))),
              patch("epub_tts_narrator.renderer.workflow.import_voice",
                    side_effect=lambda b, voice, audio, text, c, e, v: imported.append((voice, audio, text))
                    or {"reference_audio": f"/new/{voice}.wav"})):
            counts = voice_transfer.import_voices(new, {}, {}, Path("/books/old"), "engine", "verifier",
                                                  output=lines.append)
        self.assertEqual(sorted(imported), [("n1", "/v/mara-reference.wav", "Hi."),
                                            ("vn", "/v/narrator-reference.wav", "Once.")])
        self.assertEqual(traits, [("n1", "Light Irish accent.")])
        self.assertEqual(counts, {"imported": 2, "kept": 2, "missing": 0, "unmatched": 0})
        self.assertIn("  kept     Tom: this book already chose its voice", lines)
        # The same recording inherited through the series needs no copy.
        self.assertIn("  kept     Ned: already uses that voice (inherited)", lines)


if __name__ == "__main__":
    unittest.main()
