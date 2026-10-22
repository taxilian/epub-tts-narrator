"""Speech tags must not inherit the character voice around them."""

from copy import deepcopy
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.narration_boundaries import narration_tag_ranges, repair_narration_boundaries
from epub_tts_narrator.performance import apply_bracket_delivery
from epub_tts_narrator.pipeline import segment
from test_contracts import valid_document


def fixture(text, *, kind="dialogue", basis="explicit"):
    document = valid_document()
    document["cast"]["characters"][0]["evidence"] = []
    document["cast"]["voices"][1]["traits"] = []
    document["source"]["blocks"][0]["text"] = text
    document["segments"][0:2] = [segment(document["source"], "b1", 0, text,
                                        document["cast"]["roles"][1], kind=kind, basis=basis)]
    return document


class NarrationBoundaryTests(unittest.TestCase):
    def test_middle_tag_with_straight_or_curly_quotes_gets_narrator_voice(self):
        for text in ('"Wow, this is amazing", she said, "Why should we do that again?"',
                     '“Wow, this is amazing”, she said, “Why should we do that again?”'):
            with self.subTest(text=text):
                before = fixture(text)
                original = deepcopy(before)
                messages = []
                after = repair_narration_boundaries(before, progress=messages.append)
                pieces = [item for item in after["segments"] if item["block_id"] == "b1"]
                self.assertEqual([item["role_id"] for item in pieces], ["mara", "narrator", "mara"])
                self.assertEqual(pieces[1]["text"], ', she said, ')
                self.assertEqual(pieces[1]["voice_id"], "vn")
                self.assertEqual("".join(item["text"] for item in pieces), text)
                self.assertEqual(after["source"], before["source"])
                self.assertEqual(after["cast"], before["cast"])
                self.assertEqual(before, original)
                self.assertEqual(validate_document(after), [])
                self.assertEqual(len(messages), 1)
                self.assertEqual(repair_narration_boundaries(after), after)

    def test_trailing_and_inverted_named_speech_tags(self):
        cases = [('“Hello,” Mara said softly.', ["mara", "narrator"]),
                 ('“Hello,” said Mara.', ["mara", "narrator"]),
                 ('“Hello,” the old woman said.', ["mara", "narrator"])]
        for text, expected in cases:
            with self.subTest(text=text):
                result = repair_narration_boundaries(fixture(text))
                self.assertEqual([item["role_id"] for item in result["segments"] if item["block_id"] == "b1"], expected)
                self.assertEqual(validate_document(result), [])

    def test_unquoted_reported_dialogue_is_not_reinterpreted_as_narration(self):
        for text in ('I said "Go," and then I left.',
                     'Mara said "Go." That\'s all she told me.',
                     'I said "Go," she said, "Wait."',
                     'Mara said softly, “Hello.”'):
            with self.subTest(text=text):
                document = fixture(text)
                self.assertEqual(repair_narration_boundaries(document), document)

    def test_independently_narrated_prefix_allows_later_tag_repair(self):
        text = 'Mara waited. “Hello,” she said, “Again.”'
        document = fixture(text)
        start = text.index("“")
        narrator, mara = document["cast"]["roles"][:2]
        document["segments"][:1] = [segment(document["source"], "b1", 0, text[:start], narrator),
                                   segment(document["source"], "b1", start, text[start:], mara, kind="dialogue")]
        result = repair_narration_boundaries(document)
        self.assertEqual([item["role_id"] for item in result["segments"] if item["block_id"] == "b1"],
                         ["narrator", "mara", "narrator", "mara"])

    def test_boundary_whitespace_in_dialogue_is_not_a_repair(self):
        text = '“Hello,” she said, “Again.”'
        document = fixture(text)
        start, end = narration_tag_ranges(text)[0]
        narrator, mara = document["cast"]["roles"][:2]
        document["segments"][:1] = [segment(document["source"], "b1", 0, text[:start + 1], mara, kind="dialogue"),
                                   segment(document["source"], "b1", start + 1, text[start + 1:end], narrator),
                                   segment(document["source"], "b1", end, text[end:], mara, kind="dialogue")]
        self.assertEqual(repair_narration_boundaries(document), document)

    def test_already_correct_splits_are_identical_and_detached(self):
        document = valid_document()
        document["readiness"] = {"custom": "preserve even on no-op"}
        messages = []
        result = repair_narration_boundaries(document, progress=messages.append)
        self.assertEqual(result, document)
        self.assertIsNot(result, document)
        self.assertIsNot(result["source"], document["source"])
        self.assertEqual(messages, [])

    def test_quotes_inside_speech_and_apostrophes_do_not_become_narration(self):
        cases = [
            '"You said Earthling. Not human."',
            '“She said that yesterday, and I don’t agree.”',
            '“He told me ‘Go,’ she said, ‘and wait.’”',
            '"He told me \'Go,\' she said, \'and wait.\'"',
            '“A quotation with "mixed typography," she said, "stays unchanged.”',
            '“Unfinished,” she said, “still unfinished',
            '“Nested “quotation,” she said, “is uncertain.””',
            'No quotation, she said, but we should leave it alone.',
        ]
        for text in cases:
            with self.subTest(text=text):
                document = fixture(text)
                self.assertEqual(repair_narration_boundaries(document), document)
        self.assertEqual(narration_tag_ranges('“I don’t agree,” she said, “with Mara’s answer.”'), [(16, 27)])

    def test_user_thought_letter_and_restored_source_are_protected(self):
        text = '“Hello,” Mara said, “Again.”'
        for kind, basis in [("dialogue", "user"), ("thought", "context"), ("letter", "context")]:
            with self.subTest(kind=kind, basis=basis):
                document = fixture(text, kind=kind, basis=basis)
                self.assertEqual(repair_narration_boundaries(document), document)
        document = fixture(text)
        document["segments"][0]["source_restoration"] = {"id": "preserve-source-review"}
        self.assertEqual(repair_narration_boundaries(document), document)

    def test_continuing_multi_paragraph_quote_is_not_reinterpreted(self):
        document = fixture('“Here is a story without its closing quote.')
        second = '“Hello,” she said, “and walked away.”'
        document["source"]["blocks"][1]["text"] = second
        document["segments"][1] = segment(document["source"], "b2", 0, second,
                                             document["cast"]["roles"][1], kind="dialogue")
        self.assertEqual(repair_narration_boundaries(document), document)

    def test_ambiguous_narration_leaves_only_tag_unassigned(self):
        document = fixture('“Hello,” Mara said, “Again.”')
        document["cast"]["roles"].append({"id": "author", "kind": "narration", "character_id": None,
                                            "voice_id": "vn", "description": "Author notes."})
        block = deepcopy(document["source"]["blocks"][1])
        block.update(id="b3", text="Author narration.")
        document["source"]["blocks"].append(block)
        document["source"]["chapters"][0]["block_ids"].append("b3")
        document["segments"].append(segment(document["source"], "b3", 0, block["text"],
                                               document["cast"]["roles"][-1]))
        result = repair_narration_boundaries(document)
        tag = result["segments"][1]
        self.assertIsNone(tag["role_id"])
        self.assertIsNone(tag["voice_id"])
        self.assertEqual(tag["kind"], "narration")
        self.assertEqual(tag["attribution"]["basis"], "unresolved")
        self.assertEqual(tag["attribution"]["candidates"], ["author", "narrator"])
        self.assertEqual([item["role_id"] for item in result["segments"][:3]], ["mara", None, "mara"])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(repair_narration_boundaries(result), result)

    def test_same_block_narration_wins_over_other_scopes_in_chapter(self):
        document = fixture('“Hello,” Mara said, “Again,” she added.')
        text = document["source"]["blocks"][0]["text"]
        start, end = narration_tag_ranges(text, names=["Mara"])[0]
        document["cast"]["roles"].append({"id": "author", "kind": "narration", "character_id": None,
                                            "voice_id": "vn", "description": "Author notes."})
        narrator, mara = document["cast"]["roles"][:2]
        document["segments"][:1] = [segment(document["source"], "b1", 0, text[:start], mara, kind="dialogue"),
                                   segment(document["source"], "b1", start, text[start:end], narrator),
                                   segment(document["source"], "b1", end, text[end:], mara, kind="dialogue")]
        document["segments"][-1]["role_id"] = "author"
        result = repair_narration_boundaries(document)
        self.assertEqual([item["role_id"] for item in result["segments"] if item["block_id"] == "b1"],
                         ["mara", "narrator", "mara", "narrator"])

    def test_delivery_is_reset_only_for_tag_and_brackets_stay_exact(self):
        document = fixture('“Use [Fireball],” Mara whispered, “again.”')
        document["segments"][0]["delivery"] = {
            "direction": "Whisper", "evidence": [{"block_id": "b1", "quote": "whispered"}]}
        document = apply_bracket_delivery(document)
        result = repair_narration_boundaries(document)
        tag = next(item for item in result["segments"] if "Mara whispered" in item["text"])
        self.assertEqual(tag["role_id"], "narrator")
        self.assertEqual(tag["delivery"], {"direction": "", "evidence": []})
        bracketed = next(item for item in result["segments"] if "performance" in item)
        self.assertEqual(bracketed["spoken_text"], "Fireball")
        self.assertEqual(bracketed["role_id"], "mara")
        self.assertEqual(bracketed["delivery"]["direction"], "Whisper")
        self.assertEqual(validate_document(result), [])
        self.assertEqual(repair_narration_boundaries(result), result)

    def test_repeated_tags_have_unique_source_offsets_and_audit_records(self):
        text = '“One,” she said, “two,” she said, “three.”'
        result = repair_narration_boundaries(fixture(text))
        pieces = [item for item in result["segments"] if item["block_id"] == "b1"]
        self.assertEqual([item["role_id"] for item in pieces], ["mara", "narrator", "mara", "narrator", "mara"])
        self.assertEqual(len(result["provenance"]["narration_boundaries"]["repairs"]), 2)
        self.assertEqual(len({item["id"] for item in pieces}), 5)
        self.assertEqual("".join(item["text"] for item in pieces), text)
        self.assertEqual(validate_document(result), [])


if __name__ == "__main__":
    unittest.main()
