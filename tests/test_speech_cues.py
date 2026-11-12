"""The narration beside a line says how its own speaker says it."""

import unittest

from epub_tts_narrator.speech_cues import line_cues, speakers, tone_direction


CAST = {
    "characters": [{"id": "c1", "name": "Faldor", "aliases": ["the farmer"]},
                   {"id": "c2", "name": "Silk", "aliases": ["Prince Kheldar"]},
                   {"id": "c3", "name": "Polgara", "aliases": ["Aunt Pol"]}],
    "voices": [{"id": "v1", "description": "Male, sounds about fifty."},
               {"id": "v2", "description": "Male, quick and wry."},
               {"id": "v3", "description": "Female, low and commanding."}],
    "roles": [{"id": "narrator_main", "kind": "narration", "voice_id": "vn"},
              {"id": "faldor", "kind": "dialogue", "character_id": "c1", "voice_id": "v1"},
              {"id": "silk", "kind": "dialogue", "character_id": "c2", "voice_id": "v2"},
              {"id": "pol", "kind": "dialogue", "character_id": "c3", "voice_id": "v3"}],
}
PEOPLE = speakers(CAST)


def block(*pieces):
    """("text", role or None) pieces -> (block text, rows); None is narration."""
    rows, start = [], 0
    for text, role in pieces:
        rows.append({"start": start, "end": start + len(text), "text": text,
                     "kind": "narration" if role is None else "dialogue", "role_id": role or "narrator_main"})
        start += len(text)
    return "".join(text for text, _ in pieces), rows


def cues(*pieces):
    text, rows = block(*pieces)
    found = line_cues(rows, text, PEOPLE)
    return [found.get(id(row)) for row in rows if row["kind"] != "narration"]


def cue(*pieces):
    found = cues(*pieces)[0]
    assert found is not None
    return found


class SpeechCueTests(unittest.TestCase):
    def test_a_tag_or_beat_about_the_speaker_gives_the_sound(self):
        self.assertEqual(cue(('"Robbers?"', "faldor"), (" Faldor laughed.", None))["sound"], "laugh")
        self.assertEqual(cue(("Silk chuckled. ", None), ('"Durnik," he said.', "silk"))["sound"], "chuckle")
        self.assertEqual(cue(('"Well,"', "pol"), (" she sighed.", None))["sound"], "sigh")
        self.assertEqual(cue(('"Ahem."', "faldor"), (" The farmer cleared his throat.", None))["sound"],
                         "throat_clear")

    def test_someone_else_s_laugh_negations_and_mismatched_pronouns_give_nothing(self):
        self.assertEqual(cues(("Faldor laughed. ", None), ('"Hush," Silk said.', "silk")), [None])
        self.assertIsNone(cue(('"No,"', "silk"), (" he said without laughing.", None))["sound"])
        self.assertEqual(cues(('"No."', "silk"), (" She laughed.", None)), [None])
        self.assertEqual(cues(('"No."', "silk"), (" Then the dog barked.", None)), [None])

    def test_one_beat_between_two_lines_belongs_to_the_speaker_it_is_about(self):
        found = cues(('"Oh, no!"', "faldor"), (" Faldor laughed. ", None), ('"Not today."', "faldor"))
        assert found[0] is not None
        self.assertEqual(found[0]["sound"], "laugh")
        self.assertIsNone(found[1])
        found = cues(('"Look,"', "faldor"), (" Silk laughed. ", None), ('"Fine."', "silk"))
        self.assertIsNone(found[0])
        assert found[1] is not None
        self.assertEqual(found[1]["sound"], "laugh")

    def test_tones_go_to_scoped_tags_or_the_direction(self):
        self.assertEqual(cue(('"Who is there?"', "silk"), (" Silk whispered.", None))["tone"], "whisper")
        self.assertEqual(cue(('"Run!"', "pol"), (" Aunt Pol shouted.", None))["tone"], "shouting")
        self.assertEqual(cue(('"Well, now,"', "faldor"), (" Faldor drawled.", None))["tone"], "slow")
        self.assertEqual(cue(('"Sleep,"', "pol"), (" she said softly.", None))["tone"], "soft")
        # A manner word on an action is not how the line is said.
        self.assertIsNone(cue(('"Hm."', "silk"), (" Silk whistled softly.", None))["tone"])
        self.assertEqual(tone_direction("angry, defiant", "shouting"), "shouting, angry, defiant")
        self.assertEqual(tone_direction("loud and furious", "shouting"), "loud and furious")
        self.assertEqual(tone_direction("", "soft"), "soft and quiet")

    def test_breathed_speech_is_soft_but_ordinary_breathing_has_no_tone(self):
        for narration in (" she breathed.", " she breathed at him.", " she said under her breath."):
            with self.subTest(narration=narration):
                self.assertEqual(cue(('"Amazing,"', "pol"), (narration, None))["tone"], "soft")
        for narration in (" she breathed deeply.", " she breathed in.", " she took a deep breath."):
            with self.subTest(narration=narration):
                self.assertIsNone(cue(('"Wait,"', "pol"), (narration, None))["tone"])

    def test_observed_remembered_and_reported_sounds_are_not_the_witness_s_cues(self):
        narrations = (
            " Silk heard a whisper behind him.",
            " Silk watched Mara laugh.",
            " Silk remembered laughing.",
            " Silk said Mara laughed.",
        )
        for narration in narrations:
            with self.subTest(narration=narration):
                found = cue(('"Listen."', "silk"), (narration, None))
                self.assertIsNone(found["sound"])
                self.assertIsNone(found["tone"])

    def test_a_possessive_subject_keeps_its_own_sound(self):
        self.assertEqual(cue(('"Listen."', "silk"), (" Silk's laugh echoed.", None))["sound"], "laugh")

    def test_another_person_s_possessive_sound_gives_nothing(self):
        self.assertIsNone(cue(('"Listen."', "silk"), (" Silk spoke of Mara's laughter.", None))["sound"])

    def test_a_perception_verb_separated_by_a_conjunction_or_comma_does_not_suppress_the_speaker_s_sound(self):
        narrations = (
            " Silk heard Mara's story and laughed.",
            " Silk, having heard the news, laughed.",
            " Silk saw the hawk and laughed.",
            " Silk had heard the story and laughed.",
        )
        for narration in narrations:
            with self.subTest(narration=narration):
                self.assertEqual(cue(('"Listen."', "silk"), (narration, None))["sound"], "laugh")

    def test_a_perception_verb_in_an_earlier_sentence_does_not_suppress_a_later_sound(self):
        found = cue((" Silk heard a whisper. ", None), ('"Listen."', "silk"), (" Silk laughed.", None))
        self.assertEqual(found["sound"], "laugh")

    def test_listening_to_another_person_s_laughter_gives_no_sound(self):
        self.assertIsNone(cue(('"Listen."', "silk"), (" Silk listened to the laughter.", None))["sound"])

    def test_unrealized_and_explicitly_absent_actions_make_no_sound(self):
        narrations = (
            " Silk almost laughed.",
            " Silk nearly laughed.",
            " Silk was about to laugh.",
            " Silk started to laugh.",
            " Silk began to laugh.",
            " Silk answered without so much as a laugh.",
            " Silk answered without any laughter.",
            " Silk answered without laughing.",
        )
        for narration in narrations:
            with self.subTest(narration=narration):
                self.assertIsNone(cue(('"No."', "silk"), (narration, None))["sound"])

    def test_silent_actions_make_no_sound_event(self):
        narrations = (
            " Silk wept silently.",
            " Silk laughed in silence.",
            " Silk laughed without a sound.",
            " Silk laughed noiselessly.",
        )
        for narration in narrations:
            with self.subTest(narration=narration):
                self.assertIsNone(cue(('"Oh."', "silk"), (narration, None))["sound"])

    def test_breathlessness_and_wheezing_as_a_manner_are_not_pant_sounds(self):
        for narration in (" Silk said breathlessly.", " Silk was breathless.", " Silk was out of breath."):
            with self.subTest(narration=narration):
                self.assertIsNone(cue(('"Yes."', "silk"), (narration, None))["sound"])
        for narration in (" Silk said, wheezing.", " Silk, wheezing, said."):
            with self.subTest(narration=narration):
                self.assertIsNone(cue(('"Yes."', "silk"), (narration, None))["sound"])
        self.assertEqual(cue(('"Yes."', "silk"), (" Silk wheezed.", None))["sound"], "pant")
        self.assertEqual(cue(('"Yes."', "silk"), (" Silk said breathlessly.", None))["tone"], "hurried")

    def test_loud_verbs_with_laughter_mean_laughing_not_screaming_or_shouting(self):
        for verb in ("howled", "roared", "bellowed"):
            with self.subTest(verb=verb):
                found = cue(('"Ha!"', "silk"), (f" Silk {verb} with laughter.", None))
                self.assertEqual((found["sound"], found["tone"]), ("laugh", None))
        self.assertEqual(cue(('"No!"', "silk"), (" Silk howled.", None))["tone"], "screaming")

    def test_an_inverted_speech_tag_owns_its_sound_or_tone(self):
        cases = ((" sighed Silk.", "sound", "sigh"),
                 (" chuckled Silk.", "sound", "chuckle"),
                 (" murmured Silk.", "tone", "soft"))
        for narration, kind, expected in cases:
            with self.subTest(narration=narration):
                self.assertEqual(cue(('"Well."', "silk"), (narration, None))[kind], expected)

    def test_a_participle_beside_the_speaker_s_speech_tag_keeps_its_sound(self):
        self.assertEqual(cue(('"Well,"', "silk"), (" Silk, laughing, said.", None))["sound"], "laugh")

    def test_tone_words_are_not_blocked_by_substrings_inside_other_words(self):
        cases = (("slow, deliberate", "soft", "soft and quiet, slow, deliberate"),
                 ("breakfast-table ease", "hurried", "hurried, quick, breakfast-table ease"),
                 ("recalled, careful", "shouting", "shouting, recalled, careful"))
        for direction, tone, expected in cases:
            with self.subTest(direction=direction, tone=tone):
                self.assertEqual(tone_direction(direction, tone), expected)

    def test_a_bare_tag_is_no_cue_and_thoughts_make_no_sounds(self):
        self.assertEqual(cues(('"Yes,"', "faldor"), (" Faldor said.", None)), [None])
        text, rows = block(('"How odd,"', "silk"), (" Silk thought, laughing.", None))
        rows[0]["kind"] = "thought"
        found = line_cues(rows, text, PEOPLE)[id(rows[0])]
        self.assertEqual((found["sound"], found["clause"]), (None, "Silk thought, laughing."))


if __name__ == "__main__":
    unittest.main()
