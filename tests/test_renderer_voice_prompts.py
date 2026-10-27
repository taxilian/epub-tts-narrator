"""Authored descriptions pass through unchanged, without automatic context."""

import unittest

from epub_tts_narrator.renderer.voice_prompts import (
    AUDIOBOOK_VOICE_CONTEXT, MAX_PROMPT_CHARACTERS, audiobook_voice_prompt,
    default_voice_prompt, voice_prompt_description,
)


LEGACY_PREFIXES = (
    AUDIOBOOK_VOICE_CONTEXT + " ",
    "Adult audiobook narrator portraying a character in a book. "
    "Create an adult performer's imitation of the following vocal qualities: ",
    "audiobook narrator portraying a character in a book: ",
    AUDIOBOOK_VOICE_CONTEXT + "  ",
)


class AudiobookVoicePromptTests(unittest.TestCase):
    def test_custom_direction_is_preserved_exactly_without_auto_context(self):
        direction = "  A thin, high rasp.\nKeep the character's urgent cadence.  "
        self.assertEqual(
            audiobook_voice_prompt(direction),
            direction,
        )

    def test_youthful_character_description_does_not_trigger_added_context(self):
        description = "A young girl's light, bright tone with crisp consonants and lively phrasing."
        prompt = default_voice_prompt({"description": description})
        self.assertEqual(prompt, description)

    def test_redundant_user_example_keeps_only_the_authored_adult_portrayal(self):
        description = (
            "An adult woman voice actor portraying a teenage girl through voice acting, "
            "with a bright forward mid-high tone, crisp consonants, and a quick, sardonic "
            "matter-of-fact cadence that drips playful disdain; General American with a "
            "light Southern rounding."
        )
        prompt = AUDIOBOOK_VOICE_CONTEXT + " " + description
        self.assertEqual(audiobook_voice_prompt(prompt), description)
        self.assertEqual(audiobook_voice_prompt(description), description)
        self.assertEqual(default_voice_prompt({"description": prompt}), description)

    def test_old_context_is_removed_once_and_still_matches_its_description(self):
        description = "A bright, youthful character with light resonance."
        for prefix in LEGACY_PREFIXES:
            with self.subTest(prefix=prefix):
                old = prefix + description
                new = audiobook_voice_prompt(old)
                self.assertEqual(new, description)
                self.assertEqual(new, audiobook_voice_prompt(description))
                self.assertEqual(voice_prompt_description(old), description)
                self.assertEqual(voice_prompt_description(new), description)
                self.assertEqual(audiobook_voice_prompt(new), new)

    def test_default_prompt_strips_historical_wrappers_without_normalizing_prose(self):
        description = "A clear,\n measured voice."
        for prefix in ("", *LEGACY_PREFIXES):
            with self.subTest(prefix=prefix):
                first = default_voice_prompt({"description": prefix + description})
                self.assertEqual(first, description)
                self.assertEqual(default_voice_prompt({"description": first}), description)

    def test_stacked_historical_wrappers_are_removed_in_one_idempotent_pass(self):
        description = "An adult performer with a light, dry rasp.\nPreserve  this spacing.  "
        for first in LEGACY_PREFIXES:
            for second in LEGACY_PREFIXES:
                with self.subTest(first=first, second=second):
                    wrapped = first + second + description
                    self.assertEqual(voice_prompt_description(wrapped), description)
                    self.assertEqual(audiobook_voice_prompt(wrapped), description)
                    self.assertEqual(default_voice_prompt({"description": wrapped}), description)
                    self.assertEqual(audiobook_voice_prompt(audiobook_voice_prompt(wrapped)), description)

    def test_known_wrappers_preserve_custom_description_bytes_without_broad_stripping(self):
        description = "A thin, high rasp.\nKeep this  spacing.  "
        for prefix in LEGACY_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertEqual(voice_prompt_description(prefix + description), description)
                self.assertEqual(audiobook_voice_prompt(prefix + description), description)
        # Longer custom whitespace is not mistaken for the exact historical
        # double-space separator, and new custom input remains byte-exact.
        for leading in ("  ", "\t", "\n"):
            custom = leading + description
            wrapped = AUDIOBOOK_VOICE_CONTEXT + " " + custom
            self.assertEqual(voice_prompt_description(wrapped), custom)
            self.assertEqual(audiobook_voice_prompt(wrapped), custom)
            self.assertEqual(audiobook_voice_prompt(custom), custom)

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

    def test_authored_context_is_not_removed_or_rewritten(self):
        for description in (
                "An adult actor portraying a young character with light resonance.",
                "Adult voice, bright and clear, with precise diction.",
                "A fictional creature with a sharp, reedy voice.",
                "The voice sounds like an adult audiobook narrator portraying a character."):
            with self.subTest(description=description):
                self.assertEqual(audiobook_voice_prompt(description), description)
                self.assertEqual(audiobook_voice_prompt(audiobook_voice_prompt(description)), description)

    def test_default_description_excludes_trait_fields_and_source_evidence(self):
        prompt = default_voice_prompt({
            "description": " A clear\n voice. ",
            "traits": [{"name": "pitch", "value": "medium-low",
                        "evidence": [{"quote": "This is a source quotation."}]}],
        })
        self.assertEqual(prompt, " A clear\n voice. ")
        self.assertNotIn("source quotation", prompt)
        self.assertEqual(audiobook_voice_prompt(prompt), prompt)

    def test_maximum_custom_direction_uses_the_complete_description_budget(self):
        direction = "x" * MAX_PROMPT_CHARACTERS
        prompt = audiobook_voice_prompt(direction)
        self.assertEqual(len(prompt), MAX_PROMPT_CHARACTERS)
        self.assertEqual(prompt, direction)
        self.assertEqual(audiobook_voice_prompt(prompt), prompt)
        self.assertEqual(default_voice_prompt({"description": direction}), direction)
        for prefix in LEGACY_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertEqual(audiobook_voice_prompt(prefix + direction), direction)

    def test_overlong_custom_direction_is_rejected_instead_of_trimmed(self):
        direction = "x" * (MAX_PROMPT_CHARACTERS + 1)
        with self.assertRaisesRegex(ValueError, "4000"):
            audiobook_voice_prompt(direction)
        with self.assertRaisesRegex(ValueError, "4000"):
            audiobook_voice_prompt(AUDIOBOOK_VOICE_CONTEXT + " " + direction)

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
