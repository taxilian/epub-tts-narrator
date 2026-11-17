"""Troubleshooting a passage blocked on a name: what it heard, its stream, and the ways out."""

import unittest

from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.history_review import HistoryReview, load_avoid
from epub_tts_narrator.renderer.take_feedback import export, load
from epub_tts_narrator.renderer.take_review import TakeReview
from test_renderer_continuity import (FFMPEG, FFPROBE, NameModel, OtherChapterExampleTests, Verifier, histories,
                                      tagged)


def taking(root, text):
    """The take folder (request hash) of the accepted take saying ``text``."""
    [path] = [path for path in (root / "work" / "clips").glob("*/accepted.json")
              if text in read_json(path)["request"]["chunk"]["spoken_text"]]
    return path.parent.name


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class HistoryReviewTests(OtherChapterExampleTests):
    test_the_first_mention_in_a_chapter_hears_an_earlier_chapter_saying_it = None
    test_a_take_heard_rearranged_or_retired_is_never_an_example = None
    test_unchanged_work_hears_the_same_example_however_the_index_lists_its_takes = None
    test_a_line_that_passed_keeps_its_example_after_the_example_is_retired = None

    def setUp(self):
        super().setUp()
        self.config["render"]["history_reuse"] = "strict"
        self.model = self.engine._model = NameModel()
        self.verifier = Verifier(self.model)

    def review(self, text):
        return HistoryReview(TakeReview(self.root / "work", self.config, self.verifier, taking(self.root, text)))

    def test_what_a_passage_heard_names_an_example_from_another_chapter(self):
        self.render(self.chapter(["The road ran on.", "Ce’Nedra objected."], "ch1", 1, "b1"), 1)
        self.render(self.chapter(["Garion rode ahead.", "Ce’Nedra laughed."], "ch2", 10, "b2"), 2)
        rows = self.review("laughed").heard()
        self.assertEqual([(row["relation"], row["text"]) for row in rows],
                         [("first turn (the voice's first line)", "Garion rode ahead."),
                          ("reading example from another chapter", "Ce’Nedra objected.")])
        self.assertTrue(all(row["in_use"] for row in rows))

    def test_a_passage_told_not_to_hear_a_take_hears_another_and_nothing_else_changes(self):
        first = self.chapter(["The road ran on.", "Ce’Nedra objected."], "ch1", 1, "b1")
        self.render(first, 1)
        second = self.chapter(["Garion rode ahead.", "Ce’Nedra laughed."], "ch2", 10, "b2")
        self.render(second, 2)
        history = self.review("laughed")
        example = [row for row in history.heard() if row["relation"].startswith("reading example")]
        self.assertEqual(history.avoid(example), 1)
        self.assertEqual(list(load_avoid(self.root / "work").values()), [{example[0]["request_hash"]}])
        made = len(self.model.calls)
        self.render(second, 2)
        # Only that passage renders again, without the example; chapter one is untouched.
        self.assertEqual(len(self.model.calls), made + 1)
        self.assertNotIn("objected", " ".join(histories(self.model)[-1]))
        self.render(first, 1)
        self.assertEqual(len(self.model.calls), made + 1)

    def test_a_stream_set_aside_from_a_take_renders_again_from_there(self):
        lines = ["The road ran on.", "Silk smiled.", "Garion rode ahead.", "Ce’Nedra laughed."]
        chapter = self.chapter(lines, "ch1", 1, "b1")
        self.render(chapter, 1)
        history = self.review("laughed")
        stream = history.stream()
        self.assertEqual([row["text"] for row in stream], lines[:3])
        self.assertEqual(stream[0]["relation"], "the voice's first line in the chapter")
        rows = history.stream_from(stream[1])
        self.assertEqual([row["text"] for row in rows], lines[1:3])
        self.assertEqual(len(history.set_aside(rows)), 2)
        made = len(self.model.calls)
        self.render(chapter, 1)
        # The two set aside and the line after them render again; the first line is kept.
        self.assertEqual([text for text, options in self.model.calls[made:]],
                         [tagged(lines[1]), tagged(lines[2]),
                          tagged("<|phoneme_start|>S EH2 N EH1 D R AH0<|phoneme_end|> laughed.")])
        # Set aside, not judged bad: no verification disagreement is exported.
        self.assertEqual({entry["verdict"] for entry in load(self.root / "work")["takes"].values()}, {"redo"})
        path, counts = export(self.root / "work")
        self.assertEqual((counts["false_rejection"], counts["false_acceptance"]), (0, 0))

    def test_the_immediate_history_is_the_newest_attempts_turns(self):
        self.render(self.chapter(["The road ran on.", "Silk smiled.", "Ce’Nedra laughed."], "ch1", 1, "b1"), 1)
        history = self.review("laughed")
        self.assertEqual([row["text"] for row in history.immediate()], ["The road ran on.", "Silk smiled."])


if __name__ == "__main__":
    unittest.main()
