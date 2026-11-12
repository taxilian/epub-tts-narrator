"""A voice's accent is decided before the voice is designed."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer import accent_readiness
from epub_tts_narrator.renderer.accent_assignment import save_assignments
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.speech_traits import save_accent, save_speech_traits


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "book"
        shutil.copytree(EXAMPLE, self.directory)
        self.bundle = prepare_bundle(load_bundle(self.directory))
        config = deepcopy(DEFAULTS)
        config["provider"] = {"model": "fixture-model"}  # as a loaded TOML file carries it
        self.config = resolve_book_config(self.bundle, config)
        self.voices = [voice["id"] for voice in self.bundle["cast"]["voices"]]

    def test_every_speaking_voice_needs_a_decision_and_any_decision_counts(self):
        first, second = self.voices[:2]
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config), self.voices)
        save_assignments(self.bundle, self.config, {first: "Welsh accent"})
        save_accent(self.bundle, self.config, second, "")  # "no cue" is a decision too
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config, [first, second]), [])

    def test_an_accent_in_traits_a_person_set_decides_it_but_other_traits_do_not(self):
        voice_id = self.voices[0]
        save_speech_traits(self.bundle, self.config, voice_id, "calm and slow")
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config, [voice_id]), [voice_id])
        save_speech_traits(self.bundle, self.config, voice_id, "Heavy Russian accent, calm and slow")
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config, [voice_id]), [])

    def test_silent_and_derived_voices_follow_their_lines_and_main_voice(self):
        base = self.voices[0]
        self.bundle["cast"]["voices"].append({"id": "voice_older", "description": "Older",
                                              "derived_from": {"voice_id": base, "kind": "aging",
                                                               "instruction": "older"}})
        self.bundle["cast"]["voices"].append({"id": "voice_silent", "description": "Says nothing"})
        self.bundle["chapters"][0]["rows"][0] = dict(self.bundle["chapters"][0]["rows"][0], voice_id="voice_older")
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config, ["voice_older", "voice_silent"]),
                         ["voice_older"])
        save_assignments(self.bundle, self.config, {base: "Welsh accent"})
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config, ["voice_older"]), [])

    def test_without_a_text_model_or_with_assignment_off_nothing_is_required(self):
        for change in ({"provider": {}}, {"casting": dict(self.config["casting"], assign_accents=False)}):
            with self.subTest(change=change):
                config = dict(self.config, **change)
                self.assertEqual(accent_readiness.undecided(self.bundle, config), [])

    def test_casting_stops_before_loading_any_model_and_names_the_command(self):
        loaded = []
        session = CastingSession(self.bundle, self.config, notify=lambda _: None,
                                 engine_factory=lambda settings: loaded.append(settings),
                                 verifier_factory=lambda *a, **k: loaded.append(a))
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ValueError, "accents .*book") as raised:
            session.generate(self.voices[0])
        self.assertIn(str(self.directory), str(raised.exception))
        self.assertEqual(loaded, [])

    def test_a_render_needs_decisions_only_for_voices_it_would_design(self):
        with self.assertRaisesRegex(ValueError, "no accent yet"):
            accent_readiness.require_before_render(self.bundle, self.config)
        save_assignments(self.bundle, self.config, {voice_id: "Welsh accent" for voice_id in self.voices})
        accent_readiness.require_before_render(self.bundle, self.config)


if __name__ == "__main__":
    unittest.main()
