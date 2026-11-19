"""The cast screen previews a group's voices together, and its murmur, without changing anything."""

from contextlib import nullcontext
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import wave

from epub_tts_narrator.renderer.casting import CastingSession

RATE = 8000


class ToneEngine:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.calls.append((text, voice["voice_id"], direction))
        frequency = 200 + 40 * int(voice["voice_id"][-1])
        samples = [round(5000 * math.sin(2 * math.pi * frequency * index / RATE)) for index in range(RATE // 2)]
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            stream.writeframes(struct.pack("<" + "h" * len(samples), *samples))
        return {}


def session(root, mode="unison", accepted=("v0", "v1", "v2")):
    profiles = {"v0": {"id": "v0", "ensemble": {"members": ["v1", "v2"], "mode": mode, "basis": "user"}},
                "v1": {"id": "v1", "member_of": "v0"}, "v2": {"id": "v2", "member_of": "v0"}}
    engine = ToneEngine()
    fake = SimpleNamespace(
        _profile=lambda voice_id: profiles[voice_id],
        store=SimpleNamespace(directory=root, load=lambda voice_id: {"voice_id": voice_id} if voice_id in accepted else None),
        bundle={"chapters": [{"rows": [{"id": "s1", "kind": "dialogue", "voice_id": "v0", "text": "“We are here,”",
                                        "start": 0, "end": 14, "block_id": "b1"}]}]},
        config={"render": {"sample_rate": RATE, "seed": 7}},
        _get_engine=lambda: engine, _locked=nullcontext,
        _draft=lambda voice_id, selected: {"reference_text": "An audition."})
    return fake, engine


def frames(path):
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes()


class EnsembleAuditionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_a_lane_mode_previews_the_groups_line_said_together(self):
        fake, engine = session(self.root)
        path = CastingSession.hear_ensemble(fake, "v0")
        self.assertIn("ensemble-audition", path.name)
        self.assertEqual({voice for _, voice, _ in engine.calls}, {"v0", "v1", "v2"})
        self.assertEqual({text for text, _, _ in engine.calls}, {"We are here,"})
        self.assertLess(frames(path), 3 * RATE // 2, "said together, not one after another")

    def test_a_one_voice_mode_plays_the_voices_one_after_another(self):
        fake, _ = session(self.root, mode="pool")
        self.assertGreater(frames(CastingSession.hear_ensemble(fake, "v0")), 3 * RATE // 2)

    def test_the_murmur_preview_is_ten_seconds_of_quiet_talk(self):
        fake, engine = session(self.root)
        path = CastingSession.hear_ensemble(fake, "v0", murmur=True)
        self.assertIn("murmur-audition", path.name)
        self.assertEqual(frames(path), 10 * RATE)
        self.assertTrue(all(direction == "talking quietly to a neighbour" for _, _, direction in engine.calls))

    def test_every_voice_needs_an_accepted_take_first(self):
        fake, _ = session(self.root, accepted=("v0", "v1"))
        with self.assertRaisesRegex(ValueError, "Accept a take"):
            CastingSession.hear_ensemble(fake, "v0")


if __name__ == "__main__":
    unittest.main()
