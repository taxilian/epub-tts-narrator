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


class SpeechCueTests(unittest.TestCase):
    def test_a_tag_or_beat_about_the_speaker_gives_the_sound(self):
        self.assertEqual(cues(('"Robbers?"', "faldor"), (" Faldor laughed.", None))[0]["sound"], "laugh")
        self.assertEqual(cues(("Silk chuckled. ", None), ('"Durnik," he said.', "silk"))[0]["sound"], "chuckle")
        self.assertEqual(cues(('"Well,"', "pol"), (" she sighed.", None))[0]["sound"], "sigh")
        self.assertEqual(cues(('"Ahem."', "faldor"), (" The farmer cleared his throat.", None))[0]["sound"],
                         "throat_clear")

    def test_someone_else_s_laugh_negations_and_mismatched_pronouns_give_nothing(self):
        self.assertEqual(cues(("Faldor laughed. ", None), ('"Hush," Silk said.', "silk")), [None])
        self.assertIsNone(cues(('"No,"', "silk"), (" he said without laughing.", None))[0]["sound"])
        self.assertEqual(cues(('"No."', "silk"), (" She laughed.", None)), [None])
        self.assertEqual(cues(('"No."', "silk"), (" Then the dog barked.", None)), [None])

    def test_one_beat_between_two_lines_belongs_to_the_speaker_it_is_about(self):
        found = cues(('"Oh, no!"', "faldor"), (" Faldor laughed. ", None), ('"Not today."', "faldor"))
        self.assertEqual(found[0]["sound"], "laugh")
        self.assertIsNone(found[1])
        found = cues(('"Look,"', "faldor"), (" Silk laughed. ", None), ('"Fine."', "silk"))
        self.assertIsNone(found[0])
        self.assertEqual(found[1]["sound"], "laugh")

    def test_tones_go_to_scoped_tags_or_the_direction(self):
        self.assertEqual(cues(('"Who is there?"', "silk"), (" Silk whispered.", None))[0]["tone"], "whisper")
        self.assertEqual(cues(('"Run!"', "pol"), (" Aunt Pol shouted.", None))[0]["tone"], "shouting")
        self.assertEqual(cues(('"Well, now,"', "faldor"), (" Faldor drawled.", None))[0]["tone"], "slow")
        self.assertEqual(cues(('"Sleep,"', "pol"), (" she said softly.", None))[0]["tone"], "soft")
        # A manner word on an action is not how the line is said.
        self.assertIsNone(cues(('"Hm."', "silk"), (" Silk whistled softly.", None))[0]["tone"])
        self.assertEqual(tone_direction("angry, defiant", "shouting"), "shouting, angry, defiant")
        self.assertEqual(tone_direction("loud and furious", "shouting"), "loud and furious")
        self.assertEqual(tone_direction("", "soft"), "soft and quiet")

    def test_a_bare_tag_is_no_cue_and_thoughts_make_no_sounds(self):
        self.assertEqual(cues(('"Yes,"', "faldor"), (" Faldor said.", None)), [None])
        text, rows = block(('"How odd,"', "silk"), (" Silk thought, laughing.", None))
        rows[0]["kind"] = "thought"
        found = line_cues(rows, text, PEOPLE)[id(rows[0])]
        self.assertEqual((found["sound"], found["clause"]), (None, "Silk thought, laughing."))


if __name__ == "__main__":
    unittest.main()
