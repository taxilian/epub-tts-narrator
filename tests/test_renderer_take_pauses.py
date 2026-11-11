"""Sentence and clause pauses inside a take get the same targets as clip joins."""

from array import array
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

from epub_tts_narrator.renderer import take_pauses
from epub_tts_narrator.renderer.assembly import compose_chapter
from epub_tts_narrator.renderer.chapter_cache import _timings, take_pause_timings
from test_renderer_assembly import clip


RATE = 10000
SENTENCE, CLAUSE = 4000, 2000  # 400ms and 200ms at 10kHz
TEXT = "One, two. Three four."


def build(path, spans):
    """Write ``[(loud, seconds), ...]`` as PCM16; loud spans stand in for words."""
    data = []
    for loud, seconds in spans:
        data += [1200 if loud else 0] * round(seconds * RATE)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(array("h", data).tobytes())
    return path


def quiet_runs(path):
    with wave.open(str(path), "rb") as stream:
        values = array("h")
        values.frombytes(stream.readframes(stream.getnframes()))
    runs, start = [], None
    for index, value in enumerate(values):
        if value == 0 and start is None:
            start = index
        elif value and start is not None:
            if start:
                runs.append(index - start)
            start = None
    return runs


# Model pauses: a long comma pause, a sentence join with almost no pause
# (as sentence-split recovery leaves it) and a stall in the middle of a phrase.
SPANS = [(False, .1), (True, .3), (False, .9), (True, .3), (False, .04),
         (True, .3), (False, .8), (True, .3), (False, .1)]
# Whisper often stretches a word across the following pause.
WORDS = [{"word": " One,", "start": .1, "end": 1.3}, {"word": " two.", "start": 1.3, "end": 1.6},
         {"word": " Three", "start": 1.64, "end": 1.94}, {"word": " four.", "start": 2.74, "end": 3.04}]


class TextBoundaryTests(unittest.TestCase):
    def test_sentence_and_clause_punctuation_but_not_abbreviations_or_ranges(self):
        text = "Mr. Smith arrived. J. R. Tolkien wrote, and then—suddenly—stopped. Pages 10–20 matter… Next."
        kinds = [(text[:offset].split()[-1], kind) for offset, kind in take_pauses.text_boundaries(text)]
        self.assertEqual(kinds, [("arrived.", "sentence"), ("wrote,", "clause"), ("then—", "clause"),
                                 ("then—suddenly—", "clause"), ("then—suddenly—stopped.", "sentence"), ("matter…", "sentence")])

    def test_closing_quotes_stay_with_their_punctuation(self):
        text = "“Go,” she said. “Now!” Then silence."
        self.assertEqual([(text[:offset], kind) for offset, kind in take_pauses.text_boundaries(text)],
                         [("“Go,”", "clause"), ("“Go,” she said.", "sentence"),
                          ("“Go,” she said. “Now!”", "sentence")])


class BoundaryEditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = build(self.root / "take-with-uneven-pauses.wav", SPANS)

    def edited_runs(self, edits):
        """Interior quiet lengths after applying the edits to the take."""
        runs, position, shift = [], 0, {}
        for edit in edits:
            shift[edit["source_frame"]] = edit.get("insert_frames", 0) - edit.get("remove_frames", 0)
        boundaries = [1000, 4000, 13000, 16000, 16400, 19400, 27400, 30400]
        for start, end in zip(boundaries[1::2], boundaries[2::2]):
            runs.append(end - start + sum(change for frame, change in shift.items() if start <= frame < end))
        return runs

    def test_every_located_boundary_gets_its_target_and_stalls_are_capped(self):
        edits, summary = take_pauses.boundary_edits(
            self.source, TEXT, WORDS, sentence_frames=SENTENCE, clause_frames=CLAUSE)
        self.assertEqual(self.edited_runs(edits), [CLAUSE, SENTENCE, CLAUSE])
        self.assertEqual((summary["adjusted"], summary["unlocated"], summary["capped"]), (2, 0, 1))
        take_pauses.validate_edits(edits, 0, 31400)

    def test_without_word_timings_only_long_quiet_is_shortened(self):
        edits, summary = take_pauses.boundary_edits(
            self.source, TEXT, [], sentence_frames=SENTENCE, clause_frames=CLAUSE)
        self.assertEqual(self.edited_runs(edits), [SENTENCE, 400, SENTENCE])
        self.assertEqual((summary["unlocated"], summary["capped"]), (2, 2))

    def test_zero_targets_leave_the_take_as_generated(self):
        self.assertEqual(take_pauses.boundary_edits(
            self.source, TEXT, WORDS, sentence_frames=0, clause_frames=0)[0], [])

    def test_mismatched_words_never_move_a_pause_onto_other_words(self):
        words = [dict(word, word=" Elsewhere") for word in WORDS]
        edits, summary = take_pauses.boundary_edits(
            self.source, TEXT, words, sentence_frames=SENTENCE, clause_frames=CLAUSE)
        self.assertEqual(summary["unlocated"], 2)
        self.assertEqual(self.edited_runs(edits), [SENTENCE, 400, SENTENCE])


class ChapterAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = build(self.root / "take-with-uneven-pauses.wav", SPANS)
        self.original = self.source.read_bytes()

    def compose(self, name, **options):
        return compose_chapter([clip(self.source, text=TEXT)], self.root / name, sample_rate=RATE,
                               sentence_pause_ms=400, clause_pause_ms=200, **options)

    def test_chapter_pauses_inside_the_take_match_the_targets(self):
        timings = {"b1:0:6:0:21": {"expected_text": TEXT, "words": WORDS}}
        result = self.compose("chapter-normalized.wav", normalize_take_pauses=True, take_timings=timings)
        self.assertEqual(quiet_runs(result["wav_path"])[:3], [CLAUSE, SENTENCE, CLAUSE])
        self.assertEqual(self.source.read_bytes(), self.original)
        segment = result["segments"][0]
        self.assertEqual(len(segment["pause_edits"]), 3)
        _timings(result)  # the saved chapter's frame accounting stays checkable

    def test_disabled_or_untimed_takes_keep_their_internal_pauses(self):
        for name, options in (("chapter-disabled.wav", {"normalize_take_pauses": False,
                                                         "take_timings": {"b1:0:6:0:21": {
                                                             "expected_text": TEXT, "words": WORDS}}}),
                              ("chapter-default.wav", {})):
            result = self.compose(name, **options)
            self.assertEqual(quiet_runs(result["wav_path"])[:3], [9000, 400, 8000])
            self.assertNotIn("pause_edits", result["segments"][0])

    def test_untimed_takes_still_have_long_quiet_capped(self):
        result = self.compose("chapter-untimed.wav", normalize_take_pauses=True, take_timings={})
        self.assertEqual(quiet_runs(result["wav_path"])[:3], [SENTENCE, 400, SENTENCE])

    def test_saved_timing_must_account_for_every_edit(self):
        timings = {"b1:0:6:0:21": {"expected_text": TEXT, "words": WORDS}}
        result = self.compose("chapter-checked.wav", normalize_take_pauses=True, take_timings=timings)
        result["segments"][0]["pause_edits"][0]["remove_frames"] += 1
        with self.assertRaises(ValueError):
            _timings(result)


class TimingSourceTests(unittest.TestCase):
    def test_verification_words_win_and_saved_alignments_fill_gaps(self):
        acceptance = [{"chunk_id": "a", "report": {"expected_text": "A.", "words": [{"word": "A."}]}},
                      {"chunk_id": "b", "report": {"expected_text": "B.", "words": []}}]
        clips = [{"chunk_id": "b", "pause_alignment": {"expected_text": "B.", "words": ["aligned"]}}]
        self.assertEqual(take_pause_timings(acceptance, clips),
                         {"a": {"expected_text": "A.", "words": [{"word": "A."}]},
                          "b": {"expected_text": "B.", "words": ["aligned"]}})

    def test_recognition_only_for_takes_without_timings_and_failures_keep_the_take(self):
        clips = [{"chunk_id": "timed", "wav_path": "timed.wav"},
                 {"chunk_id": "untimed", "wav_path": "untimed.wav"},
                 {"chunk_id": "unrecognized", "wav_path": "unrecognized.wav"}]
        acceptance = [{"chunk_id": "timed", "report": {"expected_text": "Ant. Bee.", "words": [{"word": "Ant."}]}},
                      {"chunk_id": "untimed", "report": {"expected_text": "Cat. Dog.", "words": []}},
                      {"chunk_id": "unrecognized", "report": {"expected_text": "Eel. Fox.", "words": []}}]
        calls = []

        def recognize(path, verifier, cache_dir):
            calls.append(path)
            if path == "unrecognized.wav":
                raise ValueError("Recognized word timings are malformed")
            return [{"word": " Cat.", "start": 0, "end": .2}]

        with mock.patch.object(take_pauses, "_recognized_words", recognize):
            result = take_pauses.align_missing_timings(clips, acceptance, object(), "cache",
                                                       notify=lambda message: None)
        self.assertEqual(calls, ["untimed.wav", "unrecognized.wav"])
        self.assertEqual(result[1]["pause_alignment"],
                         {"expected_text": "Cat. Dog.", "words": [{"word": " Cat.", "start": 0, "end": .2}]})
        self.assertNotIn("pause_alignment", result[0])
        self.assertNotIn("pause_alignment", result[2])

    def test_recognized_words_are_cached_by_audio_and_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            take = build(root / "take-for-word-timing.wav", SPANS)
            primary = mock.Mock(identity={"backend": "mlx-whisper", "model_sha256": {"weights": "x"},
                                          "language": "en"}, model_path="model", _model_file_state={})
            primary.transcribe.return_value = {"text": "One, two.", "segments": [
                {"words": [{"word": " One,", "start": .1, "end": .4, "probability": .9}]}]}
            with mock.patch("epub_tts_narrator.renderer.censorship_alignment._primary_verifier",
                            return_value=primary), \
                 mock.patch("epub_tts_narrator.renderer.censorship_alignment._serialized_verifier",
                            return_value=None):
                first = take_pauses._recognized_words(take, object(), root / "timings")
                second = take_pauses._recognized_words(take, object(), root / "timings")
            self.assertEqual(first, [{"word": " One,", "start": .1, "end": .4}])
            self.assertEqual(second, first)
            primary.transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
