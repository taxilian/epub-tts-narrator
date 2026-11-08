"""A character's changed voice is cast as its own voice; equipment effects stay presets."""

import unittest

from epub_tts_narrator.contracts import VOICE_EFFECT_PRESETS
from epub_tts_narrator.renderer.voice_bindings import canonical_aliases
from epub_tts_narrator.renderer.voice_effects import CHAINS, recipe
from epub_tts_narrator.renderer.voice_prompts import derived_voice_prompt


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

    def test_every_contract_preset_has_a_versioned_chain(self):
        self.assertEqual(set(CHAINS), set(VOICE_EFFECT_PRESETS))
        self.assertNotEqual(recipe("radio"), recipe("phone"))
        with self.assertRaises(ValueError):
            recipe("chorus")


if __name__ == "__main__":
    unittest.main()
