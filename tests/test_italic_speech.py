"""Speech set in italics instead of quotation marks is read by its speaker, and the tag between by the narrator."""

import unittest

from epub_tts_narrator.italic_speech import italic_misreads, split_span

TELEPATHY = "Awaken, the words said as they appeared before my eyes. For you are the hope of the night sky."
STRESS = "I told you never to come back here, child."


def block(text, *italic, block_id="b1"):
    """A source block whose ``italic`` phrases are emphasis spans (the book's italics)."""
    spans = []
    for phrase in italic:
        start = text.index(phrase)
        spans.append({"start": start, "end": start + len(phrase), "kind": "emphasis", "origin": "style"})
    return {"id": block_id, "chapter_id": "c1", "text": text, "inline_spans": spans,
            "source": {"tag": "p"}}


def pieces(value, start=0, end=None):
    end = len(value["text"]) if end is None else end
    return [value["text"][low:high] for low, high in split_span(value, start, end)]


class SplitSpanTests(unittest.TestCase):
    def test_the_tag_between_two_italic_runs_is_cut_out_for_the_narrator(self):
        self.assertEqual(pieces(block(TELEPATHY, "Awaken,", "For you are the hope of the night sky.")),
                         ["Awaken,", "For you are the hope of the night sky."])

    def test_a_short_tag_with_a_speech_or_thought_verb_is_narration_too(self):
        text = "These scripts are forbidden, I thought. You endanger me."
        self.assertEqual(pieces(block(text, "These scripts are forbidden,", "You endanger me.")),
                         ["These scripts are forbidden,", "You endanger me."])

    def test_a_roman_word_stressed_inside_italic_speech_stays_in_the_line(self):
        value = block(STRESS, "I told you", "to come back here, child.")
        self.assertEqual(pieces(value), [STRESS])

    def test_a_trailing_tag_after_the_speech_is_cut_off(self):
        text = "Don't-know, he said to my mind as his whiskers twitched."
        self.assertEqual(pieces(block(text, "Don't-know,")), ["Don't-know,"])

    def test_spans_without_italics_or_not_starting_in_italics_are_left_whole(self):
        plain = block(TELEPATHY)
        self.assertEqual(pieces(plain), [TELEPATHY])
        late = block(TELEPATHY, "For you are the hope of the night sky.")
        self.assertEqual(pieces(late), [TELEPATHY], "speech whose first words are roman is not set in italics")
        quoted = block("“Go,” she said. “Now.”", "Go")
        self.assertEqual(pieces(quoted), ["“Go,” she said. “Now.”"])


def document(text, italic, rows):
    """A one-paragraph book: Mara (dialogue role) and the narrator, ``rows`` = [(start, end, role, kind)]."""
    value = block(text, *italic)
    return {"source": {"blocks": [value], "book": {"id": "book", "title": "Book", "language": "en"}},
            "cast": {"roles": [{"id": "narrator", "kind": "narration", "voice_id": "vn"},
                               {"id": "mara", "kind": "dialogue", "voice_id": "v1", "character_id": "c1"}],
                     "characters": [{"id": "c1", "name": "Mara", "voice_id": "v1"}], "voices": []},
            "segments": [{"id": f"b1:{start}:{end}", "block_id": "b1", "start": start, "end": end,
                          "text": text[start:end], "kind": kind, "role_id": role,
                          "attribution": {"basis": "explicit"}} for start, end, role, kind in rows]}


class MisreadTests(unittest.TestCase):
    def test_a_line_that_includes_the_tag_between_its_italic_runs_is_misread(self):
        data = document(TELEPATHY, ["Awaken,", "For you are the hope of the night sky."],
                        [(0, len(TELEPATHY), "mara", "letter")])
        self.assertEqual(list(italic_misreads(data)), ["b1"])

    def test_italic_speech_the_narrator_reads_beside_its_speech_tag_is_misread(self):
        text = "Hurry, she said to my mind. I ran."
        data = document(text, ["Hurry,"], [(0, len(text), "narrator", "narration")])
        self.assertEqual(list(italic_misreads(data)), ["b1"])

    def test_a_correct_split_quoted_speech_and_narrated_thoughts_are_left_alone(self):
        split = document(TELEPATHY, ["Awaken,", "For you are the hope of the night sky."],
                         [(0, 7, "mara", "letter"), (7, 56, "narrator", "narration"),
                          (56, len(TELEPATHY), "mara", "letter")])
        self.assertEqual(italic_misreads(split), {})
        quoted = "“I said no,” she said. “Never.”"
        stressed = document(quoted, ["no", "Never"], [(0, len(quoted), "mara", "dialogue")])
        self.assertEqual(italic_misreads(stressed), {}, "italics inside quotation marks are stress")
        thought = "Where was Jan? she thought as she ran."
        narrated = document(thought, ["Where was Jan?"], [(0, len(thought), "narrator", "narration")])
        self.assertEqual(italic_misreads(narrated), {}, "a narrated thought is the book's own convention")

    def test_a_paragraph_a_person_assigned_is_skipped(self):
        data = document(TELEPATHY, ["Awaken,", "For you are the hope of the night sky."],
                        [(0, len(TELEPATHY), "mara", "letter")])
        self.assertEqual(italic_misreads(data, skip=lambda rows: True), {})


    def test_the_rerun_split_check_reads_these_paragraphs_again_once(self):
        from epub_tts_narrator.split_check import misread_blocks, record_rereads
        data = document(TELEPATHY, ["Awaken,", "For you are the hope of the night sky."],
                        [(0, len(TELEPATHY), "mara", "letter")])
        data["provenance"] = {}
        self.assertEqual(list(misread_blocks(data)), ["b1"])
        record_rereads(data["provenance"], data, ["b1"])
        self.assertEqual(misread_blocks(data), {}, "not flagged again for the same text")


class MarkupTests(unittest.TestCase):
    def test_the_models_one_span_across_the_tag_becomes_two_lines(self):
        from epub_tts_narrator.narrator_ledger import apply_delta, name_index, new_ledger
        from epub_tts_narrator.narrator_read import _markup_validator
        value = block(TELEPATHY, "Awaken,", "For you are the hope of the night sky.", block_id="b000001")
        ledger = apply_delta(new_ledger(), {"new": [["Mara", "", "f", "", 1]]}, {"b000001": TELEPATHY},
                             {1: "b000001"})
        validate = _markup_validator(ledger, name_index(ledger), [], {1: value}, {})
        result = validate({"q": [], "x": [[1, "Awaken", "night sky.", "1", "l"]]})
        self.assertEqual([TELEPATHY[item["start"]:item["end"]] for item in result["extras"]],
                         ["Awaken,", "For you are the hope of the night sky."])
        self.assertEqual({item["kind"] for item in result["extras"]}, {"l"})


if __name__ == "__main__":
    unittest.main()
