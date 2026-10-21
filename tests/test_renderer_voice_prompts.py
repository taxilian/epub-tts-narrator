"""Audiobook context belongs in design prompts, never the spoken script."""

import unittest

from epub_tts_narrator.renderer.voice_prompts import (
    AUDIOBOOK_VOICE_CONTEXT, MAX_PROMPT_CHARACTERS, audiobook_voice_prompt,
    default_voice_prompt,
)


class AudiobookVoicePromptTests(unittest.TestCase):
    def test_context_is_explicit_and_custom_direction_is_preserved_exactly(self):
        direction = "  A thin, high rasp.\nKeep the character's urgent cadence.  "
        self.assertEqual(
            audiobook_voice_prompt(direction),
            "audiobook narrator portraying a character in a book: " + direction,
        )

    def test_reapplying_context_does_not_duplicate_it(self):
        prompt = AUDIOBOOK_VOICE_CONTEXT + " A clear, measured voice."
        self.assertEqual(audiobook_voice_prompt(prompt), prompt)
        self.assertEqual(audiobook_voice_prompt(audiobook_voice_prompt(prompt)), prompt)

    def test_default_context_excludes_trait_fields_and_source_evidence(self):
        prompt = default_voice_prompt({
            "description": " A clear\n voice. ",
            "traits": [{"name": "pitch", "value": "medium-low",
                        "evidence": [{"quote": "This is a source quotation."}]}],
        })
        self.assertEqual(prompt, AUDIOBOOK_VOICE_CONTEXT + " A clear voice.")
        self.assertNotIn("source quotation", prompt)
        self.assertEqual(audiobook_voice_prompt(prompt), prompt)

    def test_maximum_custom_direction_fits_with_context(self):
        direction = "x" * (MAX_PROMPT_CHARACTERS - len(AUDIOBOOK_VOICE_CONTEXT) - 1)
        prompt = audiobook_voice_prompt(direction)
        self.assertEqual(len(prompt), MAX_PROMPT_CHARACTERS)
        self.assertTrue(prompt.endswith(direction))
        self.assertEqual(audiobook_voice_prompt(prompt), prompt)

    def test_overlong_custom_direction_is_rejected_instead_of_trimmed(self):
        direction = "x" * (MAX_PROMPT_CHARACTERS - len(AUDIOBOOK_VOICE_CONTEXT))
        with self.assertRaisesRegex(ValueError, "4000"):
            audiobook_voice_prompt(direction)
        with self.assertRaisesRegex(ValueError, "4000"):
            audiobook_voice_prompt(AUDIOBOOK_VOICE_CONTEXT + " " + "x" * 4000)

    def test_complete_description_avoids_repeating_traits_and_keeps_harsh_identity(self):
        description = ("An elderly fictional creature with a thin, reedy tone and a harsh dry rasp. "
                       "Its precise consonants and measured phrasing stay intelligible.")
        prompt = default_voice_prompt({
            "description": description,
            "traits": [
                {"name": "age", "value": "elderly"},
                {"name": "timbre", "value": "thin, reedy tone"},
                {"name": "texture", "value": "Harsh  dry\nrasp."},
                {"name": "articulation", "value": "precise consonants"},
                {"name": "cadence", "value": "measured phrasing"},
            ],
        })
        self.assertEqual(prompt, audiobook_voice_prompt(description))

    def test_trait_only_metadata_is_not_appended_or_modified(self):
        profile = {
            "description": "A deliberate speaker",
            "traits": [
                {"name": "presentation", "value": "masculine presentation"},
                {"name": "texture", "value": "gravelly rasp"},
                {"name": "timbre", "value": "GRAVELLY  RASP."},
            ],
        }
        from copy import deepcopy
        original = deepcopy(profile)
        self.assertEqual(default_voice_prompt(profile), audiobook_voice_prompt("A deliberate speaker"))
        self.assertEqual(profile, original)

    def test_large_trait_metadata_does_not_expand_the_design_request(self):
        self.assertEqual(default_voice_prompt({
            "description": "A clear voice.",
            "traits": [{"name": "timbre", "value": "rounded resonance " * 500}],
        }), audiobook_voice_prompt("A clear voice."))

    def test_overlong_default_errors_instead_of_trimming_identity(self):
        with self.assertRaisesRegex(ValueError, "shorten the description"):
            default_voice_prompt({"description": "A clear voice. " * 500, "traits": []})


if __name__ == "__main__":
    unittest.main()
