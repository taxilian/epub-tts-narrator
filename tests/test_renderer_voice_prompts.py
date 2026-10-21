"""Audiobook context belongs in design prompts, never the spoken script."""

import unittest

from audiobook_renderer.voice_prompts import (
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

    def test_default_context_preserves_traits_and_excludes_source_evidence(self):
        prompt = default_voice_prompt({
            "description": " A clear\n voice. ",
            "traits": [{"name": "pitch", "value": "medium-low",
                        "evidence": [{"quote": "This is a source quotation."}]}],
        })
        self.assertEqual(prompt, AUDIOBOOK_VOICE_CONTEXT +
                         " A clear voice.\nVoice traits: pitch: medium-low")
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

    def test_long_generated_default_keeps_context_within_character_cap(self):
        prompt = default_voice_prompt({
            "description": "A clear voice.",
            "traits": [{"name": "timbre", "value": "rounded resonance " * 500}],
        })
        self.assertTrue(prompt.startswith(AUDIOBOOK_VOICE_CONTEXT + " "))
        self.assertLessEqual(len(prompt), MAX_PROMPT_CHARACTERS)
        self.assertTrue(prompt.endswith(("rounded…", "resonance…")))
        self.assertEqual(audiobook_voice_prompt(prompt), prompt)


if __name__ == "__main__":
    unittest.main()
