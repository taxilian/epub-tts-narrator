"""Audiobook context belongs in design prompts, never the spoken script."""

import unittest

from epub_tts_narrator.renderer.voice_prompts import (
    AUDIOBOOK_VOICE_CONTEXT, MAX_PROMPT_CHARACTERS, audiobook_voice_prompt,
    default_voice_prompt, voice_prompt_description,
)


LEGACY_PREFIXES = (
    "Adult audiobook narrator portraying a character in a book. "
    "Create an adult performer's imitation of the following vocal qualities: ",
    "audiobook narrator portraying a character in a book: ",
    AUDIOBOOK_VOICE_CONTEXT + "  ",
)


class AudiobookVoicePromptTests(unittest.TestCase):
    def test_context_is_explicit_and_custom_direction_is_preserved_exactly(self):
        direction = "  A thin, high rasp.\nKeep the character's urgent cadence.  "
        self.assertEqual(
            audiobook_voice_prompt(direction),
            AUDIOBOOK_VOICE_CONTEXT + " " + direction,
        )

    def test_youthful_character_uses_audiobook_context_and_preserves_description(self):
        description = "A young girl's light, bright tone with crisp consonants and lively phrasing."
        prompt = default_voice_prompt({"description": description})
        self.assertTrue(prompt.startswith("Adult voice mimicking a character that sounds like "))
        self.assertTrue(prompt.endswith(description))

    def test_old_context_is_upgraded_once_and_still_matches_its_description(self):
        description = "A bright, youthful character with light resonance."
        for prefix in LEGACY_PREFIXES:
            with self.subTest(prefix=prefix):
                old = prefix + description
                new = audiobook_voice_prompt(old)
                self.assertTrue(new.startswith(AUDIOBOOK_VOICE_CONTEXT + " "))
                self.assertEqual(new, audiobook_voice_prompt(description))
                self.assertEqual(voice_prompt_description(old), description)
                self.assertEqual(voice_prompt_description(new), description)
                self.assertEqual(audiobook_voice_prompt(new), new)

    def test_default_prompt_recognizes_wrappers_before_normalizing_prose(self):
        description = "A clear,\n measured voice."
        expected = AUDIOBOOK_VOICE_CONTEXT + " A clear, measured voice."
        for prefix in ("", AUDIOBOOK_VOICE_CONTEXT + " ", *LEGACY_PREFIXES):
            with self.subTest(prefix=prefix):
                first = default_voice_prompt({"description": prefix + description})
                self.assertEqual(first, expected)
                self.assertEqual(default_voice_prompt({"description": first}), expected)
                self.assertEqual(first.count(AUDIOBOOK_VOICE_CONTEXT), 1)

    def test_known_wrappers_preserve_custom_description_bytes_without_broad_stripping(self):
        description = "A thin, high rasp.\nKeep this  spacing.  "
        for prefix in (AUDIOBOOK_VOICE_CONTEXT + " ", *LEGACY_PREFIXES):
            with self.subTest(prefix=prefix):
                self.assertEqual(voice_prompt_description(prefix + description), description)
                self.assertEqual(audiobook_voice_prompt(prefix + description),
                                 AUDIOBOOK_VOICE_CONTEXT + " " + description)
        # Longer custom whitespace is not mistaken for the exact historical
        # double-space separator, and new custom input remains byte-exact.
        for leading in ("  ", "\t", "\n"):
            custom = leading + description
            wrapped = audiobook_voice_prompt(custom)
            self.assertEqual(voice_prompt_description(wrapped), custom)
            self.assertEqual(audiobook_voice_prompt(wrapped), wrapped)

    def test_gemini_recovery_matches_all_known_wrappers_without_changing_saved_requests(self):
        from copy import deepcopy
        from epub_tts_narrator.renderer.casting import _recovery_inputs
        description = "A dry rasp with measured phrasing."
        current = {"designer": {"engine": "gemini_voice_design", "model": "fixture"},
                   "prompt": audiobook_voice_prompt(description), "nonce": "current"}
        for prefix in LEGACY_PREFIXES:
            saved = dict(current, prompt=prefix + description, nonce="saved")
            before = deepcopy(saved)
            with self.subTest(prefix=prefix):
                self.assertEqual(_recovery_inputs(saved), _recovery_inputs(current))
                self.assertEqual(saved, before)
        changed = dict(current, prompt=audiobook_voice_prompt("A bright, quick voice."))
        self.assertNotEqual(_recovery_inputs(changed), _recovery_inputs(current))

    def test_context_is_not_treated_as_a_description(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            audiobook_voice_prompt(AUDIOBOOK_VOICE_CONTEXT + " ")

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
