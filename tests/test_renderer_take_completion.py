"""A take that said the start of its passage and stopped keeps that start; only the missing words are generated."""

from copy import deepcopy
import math
from pathlib import Path
import struct
import tempfile
from threading import Lock
import unittest
import wave

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.take_completion import (
    completable, completion_point, cut_sample, tail_generator)
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, _render_chunk
from shared_fixtures import renderer_config
from test_renderer_speech_identity import paragraph_request, reference, speech_request
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone

TEXT = "Moreover, he had seen evidence of discomfort on her face. Whenever they were apart."
RATE = 44100
WORD_SECONDS, GAP_SECONDS = 0.25, 0.15


def words_report(heard, *, reasons=("text_mismatch",)):
    words, at = [], 0.0
    for word in heard.split():
        words.append({"word": " " + word, "start": at, "end": at + WORD_SECONDS, "probability": 0.99})
        at += WORD_SECONDS + GAP_SECONDS
    return {"passed": False, "reasons": list(reasons), "word_timing_status": "available",
            "transcript": heard, "words": words}


class CompletionPointTests(unittest.TestCase):
    def test_a_take_that_said_the_start_and_stopped_is_cut_after_its_last_word(self):
        point = completion_point(words_report("Moreover, he had seen evidence of discomfort on her face."), TEXT)
        self.assertEqual(TEXT[point["start"]:], "Whenever they were apart.")
        self.assertEqual((point["heard_words"], point["missing_words"]), (10, 4))
        self.assertAlmostEqual(point["cut_seconds"], 9 * (WORD_SECONDS + GAP_SECONDS) + WORD_SECONDS)

    def test_only_a_take_that_stopped_at_a_sentence_or_clause_break_is_finished(self):
        text = "He waited there; nothing came. She said, quietly, that it was over: they would leave."
        for heard, rest in (("He waited there;", "nothing came. She said, quietly, that it was over: they would leave."),
                            ("He waited there; nothing came.", "She said, quietly, that it was over: they would leave."),
                            ("He waited there; nothing came. She said, quietly, that it was over:", "they would leave."),
                            ("He waited there; nothing came. She said,", None),
                            ("He waited there; nothing came. She said, quietly, that", None)):
            with self.subTest(heard=heard):
                point = completion_point(words_report(heard), text)
                self.assertEqual(text[point["start"]:] if point else None, rest)
        # A person's truncated mark does not make a mid-sentence stop a break.
        self.assertIsNone(completion_point(words_report("He waited there; nothing came. She said,"), text, vouched=True))

    def test_a_misheard_word_in_the_start_still_counts_as_the_start(self):
        point = completion_point(words_report("Moreover, he had seen evidents of discomfort on her face."), TEXT)
        self.assertEqual(TEXT[point["start"]:], "Whenever they were apart.")

    def test_takes_that_are_not_a_stopped_start_are_left_to_ordinary_recovery(self):
        for heard, reasons in (
                ("Moreover, he had seen evidence of discomfort on her face. Whenever they were apart.",
                 ("text_mismatch",)),  # nothing missing
                ("Moreover, he had seen evidence of discomfort on her place.", ("text_mismatch",)),  # unclear end
                ("The rain fell on the town all night long on her face.", ("text_mismatch",)),
                ("Moreover, he", ("text_mismatch",)),  # too little said to keep
                ("Moreover, he had seen evidence of", ("text_mismatch",)),  # stopped mid-sentence
                ("Moreover, he had seen evidence of discomfort on her face.", ("audio_too_short",)),
        ):
            with self.subTest(heard=heard):
                self.assertIsNone(completion_point(words_report(heard, reasons=reasons), TEXT))
        unavailable = dict(words_report("Moreover, he had seen evidence of discomfort on her face."),
                           word_timing_status="unavailable")
        self.assertIsNone(completion_point(unavailable, TEXT))

    def test_only_takes_of_the_passages_own_words_are_completed(self):
        self.assertTrue(completable({"seed": 1}))
        for key in ("speech_resolution", "recovery", "exhaustion_selection"):
            self.assertFalse(completable({key: {}}))


class TailGeneratorTests(unittest.TestCase):
    def test_a_part_with_scoped_controls_gets_its_span_in_the_whole_line(self):
        calls = []

        def scoped(text, path, seed, span=None):
            calls.append((text, span))
            return {}
        scoped.needs_span = True

        def check(span):
            if span[0] == 12:
                raise ValueError("would cut a pronunciation")
        scoped.check_span = check
        tail = tail_generator("one two three four", scoped, offset=4)
        tail(8, "out.wav", 7)
        self.assertEqual(calls, [("three four", (12, 22))])
        self.assertFalse(tail.supports(8))  # the check refused that span
        self.assertTrue(tail.supports(14))
        self.assertFalse(tail.supports(0))
        self.assertIsNone(tail_generator("x", None))

    def test_the_cut_moves_past_the_last_word_to_the_next_quiet(self):
        voiced = [int(8000 * math.sin(i / 5)) for i in range(int(0.5 * RATE))]
        pcm = struct.pack(f"<{len(voiced)}h", *voiced) + bytes(2 * RATE)
        cut = cut_sample(pcm, RATE, 0.3)  # the recognizer's word end came early
        self.assertGreater(cut, int(0.5 * RATE))
        self.assertLess(cut, int(0.7 * RATE))


class TwoTurnHistoryTests(unittest.TestCase):
    def test_the_kept_start_replaces_the_latest_turn_so_history_stays_at_two(self):
        # serve refuses more than two history turns (Fish's first + latest).
        from unittest.mock import patch
        from epub_tts_narrator.renderer.continuity import ChapterContinuity, compose, continuation_turn
        voice = reference("narrator")
        chapter = ChapterContinuity({"engine": "fixture"}, {"narrator": voice})

        def entry(label, pinnable):
            return {"path": label, "pinnable": pinnable, "voices": ["narrator"], "position": None,
                    "marker": {"speakers": [voice["asset_hash"]], "text": f"<|speaker:0|> {label}"},
                    "identity": {"label": label}}
        kept = {"text": "<|speaker:0|> Moreover", "codes": [[7]] * 10, "speakers": [voice["asset_hash"]]}
        loaded = patch("epub_tts_narrator.renderer.continuity_codes.load_codes",
                       side_effect=lambda path, marker: {"codes": [[1]] * 10})
        for entries, expected in (([entry("first", True), entry("latest", False)], ["first", "Moreover"]),
                                  ([entry("latest", False)], ["Moreover"]), ([], ["Moreover"])):
            chapter.entries = entries
            with self.subTest(entries=len(entries)), loaded, chapter.transaction(speech_request(voice, TEXT)):
                with continuation_turn(kept):
                    _, history = compose("<|speaker:0|> Whenever they were apart.", [voice])
                self.assertEqual([text.split(" ", 1)[1] for text, _ in history], expected)
                _, ordinary = compose("<|speaker:0|> Whenever they were apart.", [voice])
                self.assertEqual(len(ordinary), len(entries))


class HistoryTurnTests(unittest.TestCase):
    def test_the_kept_start_keeps_its_accent_cue_and_its_codes_up_to_the_cut(self):
        import hashlib
        import json
        from epub_tts_narrator.renderer.take_completion import history_turn
        with tempfile.TemporaryDirectory() as folder:
            payload = {"version": 2, "codes": [list(range(100)) for _ in range(10)], "codebook_size": 1024,
                       "text": "<|speaker:0|> [American accent] Moreover, he had seen <|phoneme_start|>R EH1 L G Z"
                               "<|phoneme_end|> as well, whenever they were apart.",
                       "speakers": ["a" * 64], "raw_audio_sha256": "b" * 64, "context": {},
                       "request_binding": "c" * 64}
            blob = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            name = hashlib.sha256(blob).hexdigest() + ".codes.json"
            (Path(folder) / name).write_bytes(blob)
            marker = {"file": name, "sha256": hashlib.sha256(blob).hexdigest(), "codebook_size": 1024,
                      **{key: payload[key] for key in ("text", "speakers", "raw_audio_sha256", "context",
                                                       "request_binding")}}
            turn = history_turn({"continuity_codes": marker}, folder, 3000, 10000, "Moreover, he had seen Relg’s as well, ")
        self.assertEqual(turn["text"], "<|speaker:0|> [American accent] Moreover, he had seen Relg’s as well,")
        self.assertEqual([len(row) for row in turn["codes"]], [30] * 10)  # 30% of the take, 30% of its frames
        self.assertEqual(turn["speakers"], ["a" * 64])
        self.assertIsNone(history_turn({}, "unused", 1, 2, "x"))


class JoinedCodesTests(unittest.TestCase):
    def test_a_finished_take_s_codes_are_its_kept_start_then_its_tail(self):
        from epub_tts_narrator.renderer.take_completion import _joined_codes
        kept = {"codes": [[1, 2, 3, 4], [5, 6, 7, 8]], "codebook_size": 4096,
                "text": "<|speaker:0|> [Mild Yorkshire accent] One two three four.", "speakers": ["voice"]}
        tail = {"codes": [[9], [10]], "codebook_size": 4096, "text": "x", "speakers": ["voice"]}
        joined = _joined_codes(kept, 2, tail, "One two  three.")
        self.assertEqual(joined["codes"], [[1, 2, 9], [5, 6, 10]])
        self.assertEqual(joined["text"], "<|speaker:0|> [Mild Yorkshire accent] One two three.")
        self.assertIsNone(_joined_codes(kept, 2, None, "One."))
        self.assertIsNone(_joined_codes(kept, 2, {**tail, "codebook_size": 1024}, "One."))
        self.assertIsNone(_joined_codes(None, 0, tail, "One."))


class ContinuationTurnTests(unittest.TestCase):
    def test_the_kept_start_is_the_latest_history_turn_the_voice_hears(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity, compose, continuation_turn
        voice = reference("narrator")
        voices = {"narrator": voice}
        chapter = ChapterContinuity({"engine": "fixture"}, voices)
        codes = [[1, 2, 3] for _ in range(10)]
        with chapter.transaction(speech_request(voice, TEXT)):
            references, history = compose("<|speaker:0|> Whenever they were apart.", [voice])
            self.assertEqual(history, [])
            with continuation_turn({"text": "<|speaker:0|> Moreover, he had seen evidence",
                                    "codes": codes, "speakers": [voice["asset_hash"]]}):
                references, history = compose("<|speaker:0|> Whenever they were apart.", [voice])
        self.assertEqual(history, [("<|speaker:0|> Moreover, he had seen evidence", codes)])
        self.assertEqual([item["speaker"] for item in references], [0])


VOCABULARY = {}


def frequency(word):
    """Each distinct word is a tone of its own pitch, so the fixture recognizer can read it back."""
    index = VOCABULARY.setdefault(word, len(VOCABULARY))
    return 300 + 70 * index


def write_words(path, words, *, trailing=0.2):
    samples = []
    for word in words:
        tone = frequency(word)
        samples += [int(9000 * math.sin(2 * math.pi * tone * i / RATE)) for i in range(int(WORD_SECONDS * RATE))]
        samples += [0] * int(GAP_SECONDS * RATE)
    samples += [0] * int(trailing * RATE)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class StoppingEngine:
    """Speaks every word as a tone; its first take of the whole line stops after ``stop_after`` words."""

    identity = {"engine": "completion-fixture", "model": "offline-v1"}
    whole = TEXT

    def __init__(self, stop_after=None):
        self.stop_after = stop_after
        self.calls = []

    def synthesize_dialogue(self, turns, path, *, seed):
        text = "".join(turn["text"] for turn in turns)
        self.calls.append(text)
        words = text.split()
        if self.stop_after is not None and text == self.whole and len(self.calls) == 1:
            write_words(path, words[:self.stop_after], trailing=1.5)
        else:
            write_words(path, words)
        return {"fixture": True, "seed": seed}

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.calls.append(text)
        words = text.split()
        if self.stop_after is not None and text == TEXT and len(self.calls) == 1:
            write_words(path, words[:self.stop_after], trailing=1.5)
        else:
            write_words(path, words)
        return {"fixture": True, "seed": seed}


class PitchVerifier:
    """Hears each tone as its word, with real timings."""

    identity = {"backend": "pitch-fixture", "version": 1}

    def transcribe(self, path):
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            data = audio.readframes(audio.getnframes())
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        names = {index: word for word, index in VOCABULARY.items()}
        window = int(0.01 * rate)
        loud = [max(abs(value) for value in samples[i:i + window]) > 1000
                for i in range(0, len(samples) - window, window)]
        words, start = [], None
        for index, flag in enumerate(loud + [False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                piece = samples[start * window:index * window]
                crossings = sum(1 for a, b in zip(piece, piece[1:]) if (a < 0) != (b < 0))
                pitch = crossings / 2 / (len(piece) / rate)
                word = names.get(round((pitch - 300) / 70))
                if word is not None:
                    words.append({"word": " " + word, "start": start * window / rate,
                                  "end": index * window / rate, "probability": 0.99})
                start = None
        text = "".join(word["word"] for word in words).strip()
        end = words[-1]["end"] if words else 0.0
        return {"text": text, "segments": [{"text": text, "start": 0.0, "end": end, "avg_logprob": -0.1,
                                             "no_speech_prob": 0.001, "words": words}]}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for take completion")
class CompletionWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = renderer_config(max_attempts=3, sample_rate=RATE)
        self.voice = reference("completion-character")
        voice_path = self.work / "completion-character-reference.wav"
        write_tone(voice_path)
        self.voice.update(reference_audio=str(voice_path), reference_sha256=file_hash(voice_path))
        self.chunk = speech_request(self.voice, TEXT)["chunk"]
        self.messages = []

    def render(self, engine):
        return _render_chunk(self.chunk, 1, 1, {self.voice["voice_id"]: self.voice}, engine, engine.identity,
                             PitchVerifier(), self.config, self.work, {}, Lock(),
                             notify=self.messages.append, retry_budget=SpeechRetryBudget())

    def test_a_take_that_stopped_early_is_finished_with_only_its_missing_words(self):
        engine = StoppingEngine(stop_after=10)
        result, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [TEXT, "Whenever they were apart."])
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        completion = record["engine_metadata"]["completion"]
        self.assertEqual((record["attempt"], completion["source_attempt"]), (2, 1))
        self.assertEqual(completion["tail_text"], "Whenever they were apart.")
        self.assertLess(completion["kept_seconds"], 10 * (WORD_SECONDS + GAP_SECONDS) + 0.3)
        self.assertTrue(any("said the first 10 words and stopped" in line for line in self.messages))

    def test_a_completion_the_worker_refuses_falls_back_to_ordinary_takes(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPError

        class Refusing(StoppingEngine):
            def synthesize(self, text, voice, path, *, seed, direction=""):
                if text != TEXT:
                    self.calls.append(text)
                    raise FishHTTPError("Fish HTTP 400 (fixture): refused", retryable=False, status=400)
                return super().synthesize(text, voice, path, seed=seed, direction=direction)
        engine = Refusing(stop_after=10)
        _, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [TEXT, "Whenever they were apart.", TEXT])
        self.assertTrue(any("Finishing the stopped take failed" in line for line in self.messages))

    def test_a_narration_paragraph_that_stopped_keeps_its_start_across_turns(self):
        first, second = "It was still snowing lightly that evening. ", "She had just reached the point of launching herself."
        request = paragraph_request([(first, self.voice, ""), (second, self.voice, "")])
        engine = StoppingEngine(stop_after=7)
        engine.whole = first + second
        _, accepted = _render_chunk(request["chunk"], 1, 1, {self.voice["voice_id"]: self.voice}, engine,
                                    engine.identity, PitchVerifier(), self.config, self.work, {}, Lock(),
                                    notify=self.messages.append, retry_budget=SpeechRetryBudget())
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [first + second, second])

    def test_a_paragraph_of_several_voices_is_never_completed(self):
        from epub_tts_narrator.renderer.take_completion import paragraph_tail
        other = reference("someone-else")
        request = paragraph_request([("One two. ", self.voice, ""), ("Three four.", other, "")])
        self.assertIsNone(paragraph_tail(request["chunk"], [{}, {}], lambda *a, **k: {}))
        alone = paragraph_request([("One two. ", self.voice, ""), ("Three four.", self.voice, "")])
        calls = []
        tail = paragraph_tail(alone["chunk"], [{"text": "One two. "}, {"text": "Three four."}],
                              lambda turns, path, seed: calls.append(turns) or {})
        tail(4, "out.wav", 3)
        self.assertEqual(calls, [[{"text": "two. "}, {"text": "Three four."}]])
        self.assertFalse(tail.supports(0))

    def test_a_take_a_person_marked_truncated_is_finished_on_the_next_render(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPError
        from epub_tts_narrator.renderer.take_feedback import judged_bad_audio, judged_truncated, record_verdict
        from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed

        class AlwaysStops(StoppingEngine):
            def synthesize(self, text, voice, path, *, seed, direction=""):
                self.calls.append(text)
                if text != TEXT:
                    raise FishHTTPError("Fish HTTP 400 (fixture): refused", retryable=False, status=400)
                write_words(path, text.split()[:10], trailing=1.5)
                return {"fixture": True, "seed": seed}
        with self.assertRaises(SpeechVerificationFailed):
            self.render(AlwaysStops())
        directory = next(path.parent for path in (self.work / "clips").glob("*/attempt-001.json"))
        record = read_json(directory / "attempt-001.json")
        record_verdict(self.work, directory.name, {"attempt": 1, "record": record}, "truncated")
        self.assertEqual(judged_truncated(directory, directory.name), [(1, record["audio_sha256"])])
        self.assertIn(record["audio_sha256"], judged_bad_audio(directory))  # never used as it is

        engine = StoppingEngine()
        self.messages.clear()
        result, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, ["Whenever they were apart."])
        completion = read_json(Path(result["wav_path"]).parent / "accepted.json")["engine_metadata"]["completion"]
        self.assertEqual(completion["source_attempt"], 1)
        self.assertTrue(any("You marked take 1 truncated" in line for line in self.messages))

    def test_a_marked_take_that_stops_mid_sentence_is_not_finished_and_says_why(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPError
        from epub_tts_narrator.renderer.take_feedback import record_verdict
        from epub_tts_narrator.renderer.workflow import RenderBlocked, SpeechVerificationFailed

        class StopsMidSentence(StoppingEngine):
            def synthesize(self, text, voice, path, *, seed, direction=""):
                self.calls.append(text)
                if text != TEXT:
                    raise FishHTTPError("Fish HTTP 400 (fixture): refused", retryable=False, status=400)
                write_words(path, text.split()[:5], trailing=1.5)
                return {"fixture": True, "seed": seed}
        with self.assertRaises((SpeechVerificationFailed, RenderBlocked)):
            self.render(StopsMidSentence())
        directory = next(path.parent for path in (self.work / "clips").glob("*/attempt-001.json"))
        record = read_json(directory / "attempt-001.json")
        record_verdict(self.work, directory.name, {"attempt": 1, "record": record}, "truncated")
        from epub_tts_narrator.renderer.take_completion import marked_truncated_source

        def tail(start, path, seed):
            raise AssertionError("a mid-sentence take is never finished")
        tail.supports = lambda start: True
        source, unfinished = marked_truncated_source(directory, directory.name, TEXT, tail)
        self.assertIsNone(source)
        self.assertEqual(unfinished[0], 1)
        self.assertIn("stops mid-sentence", unfinished[1])

    def test_a_finished_take_that_stops_again_is_finished_from_where_it_got_to(self):
        text = "Moreover, he had seen it. There was discomfort on her face. Whenever they were apart."
        stops = iter([5, 6])  # the whole line stops after its first sentence, the first tail after its second

        class StopsTwice(StoppingEngine):
            def synthesize(self, said, voice, path, *, seed, direction=""):
                self.calls.append(said)
                words = said.split()
                write_words(path, words[:next(stops, len(words))], trailing=1.5)
                return {"fixture": True, "seed": seed}
        self.chunk = speech_request(self.voice, text)["chunk"]
        engine = StopsTwice()
        result, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [text, "There was discomfort on her face. Whenever they were apart.",
                                        "Whenever they were apart."])
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual((record["attempt"], record["engine_metadata"]["completion"]["source_attempt"]), (3, 2))

    def test_a_take_cut_off_mid_sentence_is_made_again_rather_than_finished(self):
        engine = StoppingEngine(stop_after=5)
        _, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [TEXT, TEXT])
        self.assertNotIn("completion", accepted.get("engine_metadata", {}))

    def test_a_finished_take_is_finished_again_only_if_it_got_further(self):
        from epub_tts_narrator.renderer.take_completion import got_further
        finished = {"completion": {"tail_start": 40}}
        self.assertTrue(completable(finished))
        self.assertTrue(got_further(finished, {"start": 52}))
        self.assertFalse(got_further(finished, {"start": 40}))
        self.assertTrue(got_further({}, {"start": 10}))

    def test_a_take_that_spoke_every_word_never_starts_a_completion(self):
        engine = StoppingEngine()
        _, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [TEXT])


if __name__ == "__main__":
    unittest.main()
