"""A character's changed voice is cloned from its main voice; equipment effects stay presets."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.contracts import VOICE_EFFECT_PRESETS
from epub_tts_narrator.renderer.voice_bindings import canonical_aliases
from epub_tts_narrator.renderer.voice_effects import CHAINS, recipe
from epub_tts_narrator.renderer.voice_prompts import derived_voice_prompt

from audition_fixtures import prepare_bundle, prepare_document
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import RenderBlocked, ensure_voice
from test_renderer_default_designer import GeminiDesigner
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferencePlaybackEngine, TranscriptVerifier,
)


def cast(*, derived=True):
    voices = [{"id": "v_jon", "description": "Jon.", "traits": []},
              {"id": "v_jon_alt", "description": "Jon, other role.", "traits": []}]
    if derived:
        voices[1]["derived_from"] = {"voice_id": "v_jon", "kind": "age", "instruction": "older",
                                     "evidence": [{"block_id": "b1", "quote": "older"}]}
    return {"characters": [{"id": "c_jon", "name": "Jon", "voice_id": "v_jon"}],
            "roles": [{"id": "jon", "kind": "dialogue", "character_id": "c_jon", "voice_id": "v_jon"},
                      {"id": "jon_older", "kind": "dialogue", "character_id": "c_jon", "voice_id": "v_jon_alt"}],
            "voices": voices}


class VoiceChangeTests(unittest.TestCase):
    def test_a_derived_voice_is_not_aliased_to_its_base_voice(self):
        self.assertEqual(canonical_aliases(cast()), {})
        # An ordinary alternate role voice still shares the character's reference.
        self.assertEqual(canonical_aliases(cast(derived=False)), {"v_jon_alt": "v_jon"})

    def test_the_derived_prompt_keeps_the_base_design_and_adds_only_the_change(self):
        prompt = derived_voice_prompt("A warm tenor with a Dublin accent.",
                                      {"kind": "age", "instruction": "about 40 now, rougher."})
        self.assertTrue(prompt.startswith("A warm tenor with a Dublin accent."))
        self.assertIn("(age): about 40 now, rougher.", prompt)
        self.assertIn("Keep the same accent", prompt)

    def test_voice_descriptions_never_carry_a_minor_age_or_a_sexual_reading(self):
        from epub_tts_narrator.renderer.voice_prompts import designer_prompt, safe_voice_description
        cases = {
            "Young boy's voice, about 11, clear light treble, curious.": "Light, bright voice, clear light treble, curious.",
            "Teen boy, about 14, warm cracking-light baritone.": "Character, warm cracking-light baritone.",
            "Boyish, mid-high, clear, earnest.": "Light, bright, mid-high, clear, earnest.",
            "A mischievous twelve-year-old girl with a lisp.": "A mischievous character with a lisp.",
            "An adult performer voicing a 9-year-old girl: bright, quick.": "A voice actor voicing a character: bright, quick.",
            "Sultry contralto, smoky and slow, mid-30s.": "Contralto, smoky and slow, mid-30s.",
            # Adults keep their ages and plain descriptions.
            "Gravelly old man, about 70, slow and wry.": "Gravelly old man, about 70, slow and wry.",
            "A confident adult female voice, 12 years a sailor.": "A confident adult female voice, 12 years a sailor.",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(safe_voice_description(text), expected)
        gemini = {"engine": "gemini_voice_design", "model": "fixture"}
        self.assertEqual(designer_prompt("Boy, about 9, piping.", {"engine": "elevenlabs"}), "Boy, about 9, piping.")
        self.assertEqual(designer_prompt("Boy, about 9, piping.", gemini), "Character, piping.")

    def test_every_contract_preset_has_a_versioned_chain(self):
        self.assertEqual(set(CHAINS), set(VOICE_EFFECT_PRESETS))
        self.assertNotEqual(recipe("radio"), recipe("phone"))
        with self.assertRaises(ValueError):
            recipe("chorus")


class DirectedPlaybackEngine(ReferencePlaybackEngine):
    def synthesize(self, text, voice, path, *, seed, direction=""):
        result = super().synthesize(text, voice, path, seed=seed, direction=direction)
        self.speech_calls[-1]["direction"] = direction
        return result


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting integration")
class DerivedVoiceCastingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        document = prepare_document(read_json(EXAMPLE / "reviewed.json"))
        export_bundle(document, self.root / "book")
        self.bundle = prepare_bundle(load_bundle(self.root / "book"))
        cast = self.bundle["cast"]
        mara = next(role for role in cast["roles"] if role["voice_id"] == "voice_mara")
        block = self.bundle["source"]["blocks"][0]
        evidence = {"block_id": block["id"], "text": block["text"]}
        cast["voices"].append({"id": "voice_mara_young", "description": "Mara as a small girl.", "traits": [],
                               "derived_from": {"voice_id": "voice_mara", "kind": "age",
                                                "instruction": "a small girl, high and light",
                                                "evidence": [{"block_id": evidence["block_id"],
                                                              "quote": evidence["text"]}]}})
        cast["roles"].append({"id": "mara_young", "kind": "dialogue", "character_id": mara["character_id"],
                              "voice_id": "voice_mara_young", "description": "Mara, young."})
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["gemini"]["api_key"] = "test-only-api-key"
        self.config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe", require_cover=False)
        self.config["quality"] = validate_policy({})
        self.transcripts = {}
        self.engine = DirectedPlaybackEngine(self.transcripts)
        self.designer = GeminiDesigner(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)

    def session(self):
        session = CastingSession(self.bundle, self.config, engine_factory=lambda _: self.engine,
                                 verifier_factory=lambda *_args, **_kwargs: self.verifier,
                                 designer_factory=lambda _settings: self.designer, notify=lambda _: None)
        self.addCleanup(session.close)
        return session

    def test_the_cast_menu_clones_a_changed_voice_from_its_cast_base_voice(self):
        session = self.session()
        with self.assertRaisesRegex(RenderBlocked, "cast voice_mara first"):
            session.ensure_ready("voice_mara_young")
        base = session.ensure_ready("voice_mara")
        designed = len(self.designer.calls)
        young = session.ensure_ready("voice_mara_young")
        # Cloned by the book engine from Mara's own recording, never designed afresh.
        self.assertEqual(len(self.designer.calls), designed)
        self.assertEqual(self.engine.speech_calls[-1]["voice_id"], "voice_mara")
        # The direction is storytelling wording, never a child's age or a word for a child.
        self.assertEqual(self.engine.speech_calls[-1]["direction"], "A character, high and light")
        self.assertEqual(young["creation"]["method"], "derived_clone")
        self.assertEqual(young["creation"]["derived_from"]["reference_sha256"], base["reference_sha256"])
        label = next(row["label"] for row in session.rows() if row["voice_id"] == "voice_mara_young")
        self.assertIn("cloned from", label)

    def test_rendering_creates_the_base_voice_first_and_then_clones_from_it(self):
        session = self.session()
        profile = next(voice for voice in self.bundle["cast"]["voices"] if voice["id"] == "voice_mara_young")
        config = deepcopy(session.config)
        with patch("epub_tts_narrator.renderer.voice_design.create_designer", return_value=self.designer):
            young = ensure_voice(session.store, "voice_mara_young", profile, self.engine, self.verifier,
                                 config, config["render"]["work_dir"], bundle=self.bundle, notify=lambda _: None)
        base = session.store.load("voice_mara")
        self.assertIsNotNone(base)
        self.assertEqual(len(self.designer.calls), 1)
        self.assertEqual(young["creation"]["derived_from"]["reference_sha256"], base["reference_sha256"])


if __name__ == "__main__":
    unittest.main()
