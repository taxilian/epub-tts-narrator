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
from test_renderer_speech_identity import reference, speech_request
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone

TEXT = "Moreover, he had seen evidence of discomfort on her face, whenever they were apart."
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
        self.assertEqual(TEXT[point["start"]:], "whenever they were apart.")
        self.assertEqual((point["heard_words"], point["missing_words"]), (10, 4))
        self.assertAlmostEqual(point["cut_seconds"], 9 * (WORD_SECONDS + GAP_SECONDS) + WORD_SECONDS)

    def test_a_misheard_word_in_the_start_still_counts_as_the_start(self):
        point = completion_point(words_report("Moreover, he had seen evidents of discomfort on her face."), TEXT)
        self.assertEqual(TEXT[point["start"]:], "whenever they were apart.")

    def test_takes_that_are_not_a_stopped_start_are_left_to_ordinary_recovery(self):
        for heard, reasons in (
                ("Moreover, he had seen evidence of discomfort on her face, whenever they were apart.",
                 ("text_mismatch",)),  # nothing missing
                ("Moreover, he had seen evidence of discomfort on her place.", ("text_mismatch",)),  # unclear end
                ("The rain fell on the town all night long on her face.", ("text_mismatch",)),
                ("Moreover, he", ("text_mismatch",)),  # too little said to keep
                ("Moreover, he had seen evidence of discomfort on her face.", ("audio_too_short",)),
        ):
            with self.subTest(heard=heard):
                self.assertIsNone(completion_point(words_report(heard, reasons=reasons), TEXT))
        unavailable = dict(words_report("Moreover, he had seen evidence of discomfort on her face."),
                           word_timing_status="unavailable")
        self.assertIsNone(completion_point(unavailable, TEXT))

    def test_only_takes_of_the_passages_own_words_are_completed(self):
        self.assertTrue(completable({"seed": 1}))
        for key in ("speech_resolution", "completion", "recovery", "exhaustion_selection"):
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


class ContinuationTurnTests(unittest.TestCase):
    def test_the_kept_start_is_the_latest_history_turn_the_voice_hears(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity, compose, continuation_turn
        voice = reference("narrator")
        voices = {"narrator": voice}
        chapter = ChapterContinuity({"engine": "fixture"}, voices)
        codes = [[1, 2, 3] for _ in range(10)]
        with chapter.transaction(speech_request(voice, TEXT)):
            references, history = compose("<|speaker:0|> whenever they were apart.", [voice])
            self.assertEqual(history, [])
            with continuation_turn({"text": "<|speaker:0|> Moreover, he had seen evidence",
                                    "codes": codes, "speakers": [voice["asset_hash"]]}):
                references, history = compose("<|speaker:0|> whenever they were apart.", [voice])
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

    def __init__(self, stop_after=None):
        self.stop_after = stop_after
        self.calls = []

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
        self.assertEqual(engine.calls, [TEXT, "whenever they were apart."])
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        completion = record["engine_metadata"]["completion"]
        self.assertEqual((record["attempt"], completion["source_attempt"]), (2, 1))
        self.assertEqual(completion["tail_text"], "whenever they were apart.")
        self.assertLess(completion["kept_seconds"], 10 * (WORD_SECONDS + GAP_SECONDS) + 0.3)
        self.assertTrue(any("said the first 10 words and stopped" in line for line in self.messages))

    def test_a_take_that_spoke_every_word_never_starts_a_completion(self):
        engine = StoppingEngine()
        _, accepted = self.render(engine)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(engine.calls, [TEXT])


if __name__ == "__main__":
    unittest.main()
