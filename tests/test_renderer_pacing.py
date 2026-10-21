"""Assembly controls padding without regenerating or altering accepted speech."""

from array import array
from pathlib import Path
import tempfile
import unittest
import wave

from epub_tts_narrator.renderer.assembly import compose_chapter
from epub_tts_narrator.renderer.pacing import audio_edges
from test_renderer_assembly import clip


RATE = 10000


def wav(path, data):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(array("h", data).tobytes())
    return path


def samples(path):
    with wave.open(str(path), "rb") as stream:
        result = array("h")
        result.frombytes(stream.readframes(stream.getnframes()))
    return result


class PacingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_variable_engine_padding_becomes_same_total_sentence_gap(self):
        results = []
        for index, (head, tail) in enumerate(((2000, 8000), (5000, 15000))):
            source = wav(self.root / f"source{index}.wav", [0] * head + [1200] * 1000 + [0] * tail)
            original = source.read_bytes()
            result = compose_chapter([
                clip(source, text='"Hello!"', break_after="none"),
                clip(source, text="Still here.", identifier="second", break_after="none"),
            ], self.root / f"chapter{index}.wav", sample_rate=RATE)
            output = samples(result["wav_path"])
            active = [i for i, value in enumerate(output) if value]
            self.assertEqual(active[1000] - active[999] - 1, 2200)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(result["segments"][0]["source_audio_start_frame"], head - 400)
            self.assertEqual(result["segments"][0]["source_audio_end_frame"], head + 1600)
            results.append(output)
        self.assertEqual(*results)

    def test_quiet_edges_breaths_and_internal_pause_are_preserved(self):
        content = [45] * 500 + [1200] * 1000 + [0] * 5000 + [1200] * 1000 + [45] * 500
        source = wav(self.root / "source.wav", [0] * 2000 + content + [0] * 3000)
        result = compose_chapter([clip(source)], self.root / "chapter.wav", sample_rate=RATE)
        output = samples(result["wav_path"])
        self.assertEqual(list(output[400:400 + len(content)]), content)
        # Every kept source frame has a direct original-WAV offset.
        timing = result["segments"][0]
        a, b = timing["source_audio_start_frame"], timing["source_audio_end_frame"]
        self.assertEqual(output[:timing["chapter_end_frame"]], samples(source)[a:b])

    def test_relative_threshold_does_not_trim_very_quiet_speech(self):
        source = wav(self.root / "quiet.wav", [0] * 1000 + [2] * 2000 + [0] * 1000)
        edges = audio_edges(source)
        self.assertEqual((edges["start"], edges["end"]), (600, 3600))

    def test_boundary_kinds_and_controls_do_not_stack(self):
        source = wav(self.root / "source.wav", [0] * 1000 + [1000] * 1000 + [0] * 1000)
        for text, changes, gap in (("First,", {}, 1000), ("First.", {}, 2200),
                                  ("First.", {"break_after": "paragraph"}, 2500)):
            target = self.root / f"gap{gap}.wav"
            result = compose_chapter([
                clip(source, text=text, break_after=changes.get("break_after", "none")),
                clip(source, identifier="second", break_after="none"),
            ], target, sample_rate=RATE)
            active = [i for i, value in enumerate(samples(result["wav_path"])) if value]
            self.assertEqual(active[1000] - active[999] - 1, gap)
        result = compose_chapter([
            clip(source), clip(None, text="[]", identifier="control", break_before="blank_paragraph"),
            clip(source, identifier="second", performance={"kind": "bracketed", "enter": True}),
        ], self.root / "controls.wav", sample_rate=RATE)
        active = [i for i, value in enumerate(samples(result["wav_path"])) if value]
        self.assertEqual(active[1000] - active[999] - 1, 5000)

    def test_disabled_policy_keeps_legacy_padding_and_all_silent_is_untouched(self):
        source = wav(self.root / "source.wav", [0] * 1000 + [1000] * 1000 + [0] * 1000)
        result = compose_chapter([clip(source)], self.root / "legacy.wav",
                                 sample_rate=RATE, normalize_pauses=False)
        self.assertEqual(samples(result["wav_path"])[:3000], samples(source))
        self.assertEqual(result["frames"], 5500)
        quiet = wav(self.root / "silent.wav", [0] * 1000)
        self.assertEqual(audio_edges(quiet)["end"], 1000)


if __name__ == "__main__":
    unittest.main()
