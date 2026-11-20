"""Plural speakers are guarded by their tags, never by names: flags only."""

import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.group_cues import group_warnings, subject_kind
from test_character_merge import build

CHARACTERS = [("mara", "Mara", ()), ("lena", "Lena", ()), ("guard", "burly guard", ()),
              ("chorus", "the chorus", ()), ("voice", "dry voice", ()), ("dryads", "Dryad girls", ())]


def document(paragraphs, *, groups=("chorus", "dryads")):
    result = build("guards", "Guards", CHARACTERS, paragraphs)
    for character in result["cast"]["characters"]:
        if character["id"] in groups:
            character["group"] = {"size": None, "scale": "several", "members": [], "basis": "user"}
    return result


def kinds(paragraphs, **options):
    data = document(paragraphs, **options)
    return [(warning["segment_id"], warning["kind"]) for warning in group_warnings(data)]


class SubjectTests(unittest.TestCase):
    def test_a_tags_subject_says_how_many_speak(self):
        for clause, expected in (
                ("the Murgos replied in unison.", "plural"), ("they said.", "plural"),
                ("Mara and Lena said together.", "plural"), ("several of them answered her.", "plural"),
                ("the crowd roared.", "plural"), ("said the twins, bowing.", "plural"),
                ("the two said in unison.", "plural"), ("those behind Mara exclaimed.", "plural"),
                ("Those assembled at the hall’s edge responded,", "plural"), ("these two whispered.", "plural"),
                ("one of the others observed,", "member"), ("another man roared.", "member"),
                ("someone shouted.", "member"), ("the other finished.", "member"),
                ("she said.", "singular"), ("the voice promised.", "singular"), ("Mara said.", "singular"),
                ("Mara waved.", None), ("The twins smiled.", None)):
            with self.subTest(clause):
                self.assertEqual(subject_kind(clause), expected)


class GroupWarningTests(unittest.TestCase):
    def test_a_line_several_people_say_given_to_one_ordinary_character_is_flagged(self):
        self.assertEqual(kinds([[("“As you command,” ", "guard"), ("the Murgos replied in unison.", None)]]),
                         [("b1:0", "plural_tag")])
        self.assertEqual(kinds([[("“Lift!” ", "mara"), ("Mara and Lena said together.", None)]]),
                         [("b1:0", "plural_tag")])

    def test_a_groups_line_tagged_with_another_characters_name_is_flagged(self):
        self.assertEqual(kinds([[("“Be quiet,” ", "chorus"), ("the voice promised.", None)]]),
                         [("b1:0", "singular_tag")])

    def test_one_of_them_given_to_a_named_character_is_flagged(self):
        self.assertEqual(kinds([[("“It has been such a long time,” ", "mara"), ("one of them said.", None)]]),
                         [("b1:0", "invented_member")])

    def test_ordinary_lines_groups_and_unison_rows_are_left_alone(self):
        self.assertEqual(kinds([
            [("“Hello,” ", "mara"), ("she said.", None)],
            [("“We are here,” ", "chorus"), ("they said in unison.", None)],
            [("“It looks healthy,” ", "dryads"), ("one of the others observed.", None)],
            [("“Kisses,” ", "dryads"), ("the Dryad offered.", None)],
            [("“Together!” ", ("mara", "lena")), ("they cried.", None)],
            [("“Halt,” ", "guard"), ("one of the guards said.", None)],
        ]), [])

    def test_without_a_group_the_chorus_is_an_ordinary_character_and_its_plural_tag_is_flagged(self):
        self.assertEqual(kinds([[("“We are here,” ", "chorus"), ("they said in unison.", None)]], groups=()),
                         [("b1:0", "plural_tag")])

    def test_verify_attribution_reads_a_flagged_paragraph_again(self):
        from epub_tts_narrator.attribution_verification import risk_blocks
        data = document([[("“Hello,” ", "mara"), ("she said.", None)],
                         [("“As you command,” ", "guard"), ("the Murgos replied in unison.", None)]])
        self.assertEqual(risk_blocks(data), ["b2"])

    def test_fixtures_are_valid_analyses(self):
        self.assertEqual(validate_document(document([[("“Hello,” ", "mara"), ("she said.", None)]])), [])


if __name__ == "__main__":
    unittest.main()
