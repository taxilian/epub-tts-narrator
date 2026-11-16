"""A take carrying a steady high-pitched tone is rejected, never chosen, and retired when in use."""

import importlib.util
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
import wave

from epub_tts_narrator.renderer.common import read_json, write_json

RATE = 44100


def write_speechlike(path, *, tone_hz=None, tone_level=0.02, seconds=4.0):
    """Syllables of a 140 Hz voice with broad hiss now and then, plus an optional steady tone throughout."""
    rng = random.Random(7)
    samples = []
    for index in range(int(seconds * RATE)):
        t = index / RATE
        syllable = math.sin(math.pi * ((t * 4) % 1)) ** 2  # four syllables a second
        voice = sum(math.sin(2 * math.pi * 140 * h * t) / h for h in range(1, 25)) * 0.08 * syllable
        hiss = rng.uniform(-1, 1) * 0.05 if (t * 4) % 3 < 0.4 else 0.0  # a broad "s" now and then
        value = voice + hiss
        if tone_hz:
            value += tone_level * math.sin(2 * math.pi * tone_hz * t)
        samples.append(max(-32767, min(32767, int(value * 32767))))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy (renderer environment)")
class SteadyToneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_a_steady_high_tone_is_found_and_speech_alone_is_not(self):
        from epub_tts_narrator.renderer.steady_tone import check, measure
        clean, tonal = self.root / "clean-take.wav", self.root / "tonal-take.wav"
        write_speechlike(clean)
        write_speechlike(tonal, tone_hz=7100)
        self.assertIsNone(check(clean), measure(clean))
        found = check(tonal)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["frequency_hz"], 7100, delta=150)
        self.assertGreaterEqual(found["presence"], 0.5)

    def test_a_short_take_of_one_held_sound_is_not_judged(self):
        from epub_tts_narrator.renderer.steady_tone import check, measure
        held = self.root / "held-sound-take.wav"
        write_speechlike(held, tone_hz=7100, seconds=1.0)
        self.assertIsNone(measure(held))
        self.assertIsNone(check(held))

    def test_verification_rejects_it_before_recognition(self):
        from epub_tts_narrator.renderer.quality import validate_audio
        import test_renderer_reference_consensus as consensus
        tonal = self.root / "tonal-take.wav"
        write_speechlike(tonal, tone_hz=10000)
        verifier = consensus.PrimaryVerifier("Hello there.")
        report = validate_audio(tonal, "Hello there.", verifier, {}, purpose="speech")
        self.assertEqual(report["reasons"], ["steady_tone"])
        self.assertEqual(verifier.calls, [])
        from epub_tts_narrator.renderer.steady_tone import describe
        self.assertRegex(describe(report), r"^a steady 10\.[01] kHz tone")

    def test_takes_in_use_with_a_tone_are_retired_once_and_others_kept(self):
        from epub_tts_narrator.renderer.steady_tone import SWEEP_FILENAME, VERSION, retire_tonal
        work = self.root / "render-work"

        def accepted(name, *, tone=None, metadata=None):
            folder = work / "clips" / (name * 64)[:64]
            folder.mkdir(parents=True)
            write_speechlike(folder / "take.wav", tone_hz=tone)
            write_json(folder / "accepted.json", {
                "attempt": 1, "audio_sha256": name * 64, "audio_file": "take.wav",
                "engine_metadata": metadata or {}, "request": {"kind": "speech"},
                "report": {"passed": True, "reasons": []}})
            write_json(folder / "exhaustion-selection.json", {"audit": {}})
            return folder
        tonal, clean = accepted("a", tone=7100), accepted("b")
        chosen = accepted("c", tone=7100, metadata={"manual_selection": {}})
        self.assertEqual(retire_tonal(work), [tonal])
        self.assertFalse((tonal / "accepted.json").exists())
        self.assertFalse((tonal / "exhaustion-selection.json").exists())  # its choice cannot republish it
        evidence = read_json(next(tonal.glob("steady-tone-*.json")))
        self.assertEqual(evidence["evidence"]["reason"], "steady_tone")
        for folder in (clean, chosen):
            self.assertTrue((folder / "accepted.json").exists())
        self.assertEqual(read_json(work / SWEEP_FILENAME)["version"], VERSION)
        self.assertEqual(retire_tonal(work), [])  # measured once



@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy (renderer environment)")
class NoHistoryAfterToneTests(unittest.TestCase):
    """After a take carrying a steady tone, the passage's next takes are made without chapter history."""

    def test_after_a_tonal_take_a_line_without_chosen_names_hears_no_history(self):
        from unittest.mock import patch
        from epub_tts_narrator.renderer.continuity import recovery_mode
        from epub_tts_narrator.renderer.workflow import verified_take
        import test_renderer_reference_consensus as consensus
        from test_renderer_stall_recovery import config
        directory = Path(tempfile.mkdtemp()) / "takes"
        directory.mkdir()
        self.addCleanup(__import__("shutil").rmtree, directory.parent)
        modes = []

        def generate(path, seed):
            modes.append(recovery_mode())
            write_speechlike(path, tone_hz=7100 if len(modes) == 1 else None)
            return {"seed": seed}
        with patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                   side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())):
            _, accepted = verified_take("Hello there.", generate, directory, {"fixture": "tone-history"},
                                        consensus.PrimaryVerifier("Hello there."), config(), purpose="speech",
                                        notify=lambda _: None)
        self.assertEqual(modes, [None, "none"])
        self.assertEqual(accepted["attempt"], 2)

    def test_after_two_takes_that_said_nothing_the_next_hears_no_history(self):
        import struct
        from unittest.mock import patch
        from epub_tts_narrator.renderer.continuity import recovery_mode
        from epub_tts_narrator.renderer.workflow import verified_take
        import test_renderer_reference_consensus as consensus
        from test_renderer_stall_recovery import config
        directory = Path(tempfile.mkdtemp()) / "takes"
        directory.mkdir()
        self.addCleanup(__import__("shutil").rmtree, directory.parent)
        modes = []

        def generate(path, seed):
            modes.append(recovery_mode())
            if len(modes) <= 2:  # near silence
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                    audio.writeframes(struct.pack("<h", 3) * RATE * 2)
            else:
                write_speechlike(path)
            return {"seed": seed}
        with patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                   side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())):
            _, accepted = verified_take("Hello there.", generate, directory, {"fixture": "silent-history"},
                                        consensus.PrimaryVerifier("Hello there."), config(max_attempts=3),
                                        purpose="speech",
                                        notify=lambda _: None)
        self.assertEqual(modes, [None, None, "none"])
        self.assertEqual(accepted["attempt"], 3)

    def test_the_reading_rung_keeps_only_turns_saying_the_line_s_readings(self):
        from epub_tts_narrator.renderer import continuity
        from epub_tts_narrator.renderer.history_echo import apply_recovery
        voice = {"asset_hash": "n"}
        chapter = type("Chapter", (), {"voices_by_asset": {"n": voice}})()
        takes = [{"marker": {"speakers": ["n"], "text": "<|speaker:0|> first"}, "path": "a", "readings": frozenset()},
                 {"marker": {"speakers": ["n"], "text": "<|speaker:0|> Marendra"}, "path": "b",
                  "readings": frozenset({"marendra"})}]
        from unittest.mock import patch

        def composed(mode, readings):
            token = continuity._CURRENT.set({"chapter": chapter, "takes": takes, "readings": readings})
            try:
                with continuity.history_recovery(mode):
                    return continuity.compose("<|speaker:0|> line", [voice])[1]
            finally:
                continuity._CURRENT.reset(token)
        with patch("epub_tts_narrator.renderer.continuity_codes.load_codes", return_value={"codes": [[1]]}):
            history = composed("reading", frozenset({"marendra"}))
            everything = composed(None, frozenset({"marendra"}))
            none = composed("reading", frozenset())
        self.assertEqual([text for text, _ in history], ["<|speaker:0|> Marendra"])
        self.assertEqual(len(everything), 2)
        self.assertEqual(apply_recovery("reading", "t", history, (), kind=None)[0], "reading")
        self.assertEqual(none, [])
        self.assertEqual(apply_recovery("reading", "t", none, (), kind=None)[:3], ("none", "t", []))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy (renderer environment)")
class FilteredFallbackTests(unittest.TestCase):
    """When every take of a passage carries the tone, the faintest is filtered and used if its words verify."""

    def setUp(self):
        from test_renderer_workflow import FFMPEG
        if not FFMPEG:
            self.skipTest("FFmpeg required")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "takes"
        self.directory.mkdir()
        self.request = {"kind": "speech", "chunk": {"spoken_text": "Hello there."}}
        self.config = {"quality": {}, "render": {"ffmpeg": FFMPEG}}

    def take(self, number, level):
        from epub_tts_narrator.renderer.common import digest
        from epub_tts_narrator.renderer.steady_tone import measure
        name = f"attempt-{number:03d}-fixture-take.wav"
        write_speechlike(self.directory / name, tone_hz=7100, tone_level=level)
        write_json(self.directory / f"attempt-{number:03d}.json", {
            "request_hash": digest(self.request), "request": self.request, "attempt": number, "seed": number,
            "audio_file": name, "audio_sha256": "f" * 64, "engine_metadata": {"seed": number},
            "report": {"passed": False, "reasons": ["steady_tone"],
                       "steady_tone": measure(self.directory / name)}})

    def test_the_faintest_tonal_take_is_filtered_verified_and_used(self):
        from epub_tts_narrator.renderer.steady_tone import MARKER, filtered_fallback
        import test_renderer_reference_consensus as consensus
        self.take(1, 0.04)
        self.take(2, 0.015)  # fainter
        said = []
        path, record = filtered_fallback(self.directory, self.request, "Hello there.",
                                         consensus.PrimaryVerifier("Hello there."), self.config, notify=said.append)
        self.assertEqual(record["attempt"], 2)
        self.assertTrue(record["report"]["passed"])
        self.assertIn(7100, [round(hz, -2) for hz in record["engine_metadata"][MARKER]["notches_hz"]])
        self.assertTrue(path.name.endswith("-tone-filtered.wav"))
        self.assertEqual(read_json(self.directory / "accepted.json")["audio_file"], path.name)
        self.assertTrue((self.directory / "attempt-002-fixture-take.wav").exists())  # the original is kept
        self.assertTrue(any("filtered out" in line for line in said))

    def test_when_the_clean_takes_all_misread_a_name_a_tonal_take_is_filtered_and_heard(self):
        from unittest.mock import patch
        from epub_tts_narrator.renderer import clip_recovery
        import test_renderer_reference_consensus as consensus
        self.take(1, 0.02)
        misread = ValueError("Every usable retained take misread a name whose pronunciation you chose (x)")
        with patch.object(clip_recovery, "select_retained", side_effect=misread):
            path, record = clip_recovery.select_or_filter(
                self.directory, self.request, None, {}, "Hello there.",
                consensus.PrimaryVerifier("Hello there."), self.config, notify=lambda _: None)
        self.assertTrue(record["report"]["passed"])
        self.assertTrue(path.name.endswith("-tone-filtered.wav"))
        silent = ValueError("Every retained take went silent before finishing its text")
        with patch.object(clip_recovery, "select_retained", side_effect=silent), \
                self.assertRaisesRegex(ValueError, "went silent"):
            clip_recovery.select_or_filter(self.directory, self.request, None, {}, "Hello there.",
                                           consensus.PrimaryVerifier("Hello there."), self.config,
                                           notify=lambda _: None)

    def test_a_filtered_take_whose_words_fail_is_not_used(self):
        from epub_tts_narrator.renderer.steady_tone import filtered_fallback
        import test_renderer_reference_consensus as consensus
        self.take(1, 0.03)
        result = filtered_fallback(self.directory, self.request, "Hello there.",
                                   consensus.PrimaryVerifier("Goodbye now."), self.config, notify=lambda _: None)
        self.assertIsNone(result)
        self.assertFalse((self.directory / "accepted.json").exists())

    def test_a_filtered_take_is_validated_as_the_take_it_came_from(self):
        from unittest.mock import patch
        from epub_tts_narrator.renderer.exhaustion_selection import validate_record
        seen = []
        record = {"request": {"kind": "speech"}, "audio_sha256": "b" * 64,
                  "engine_metadata": {"synthesis_text": "x", "clip_repair": {"audio_sha256": "a" * 64},
                                      "tone_filter": {"version": 1, "source_attempt": 4,
                                                      "source_audio_sha256": "a" * 64}}}
        with patch("epub_tts_narrator.renderer.clip_repair_contract.confirm_plan_metadata",
                   side_effect=lambda request, metadata: seen.append(sorted(metadata))):
            validate_record(record)  # the repair's proof matches the source audio, not the filtered audio
        self.assertEqual(seen, [["clip_repair", "synthesis_text"]])
        with self.assertRaisesRegex(ValueError, "unknown record"):
            validate_record({**record, "engine_metadata": {"tone_filter": {"version": 2}}})

    def test_a_filtered_take_is_never_chapter_history(self):
        from epub_tts_narrator.renderer.continuity import _NOT_HISTORY
        from epub_tts_narrator.renderer.steady_tone import MARKER
        self.assertIn(MARKER, _NOT_HISTORY)


if __name__ == "__main__":
    unittest.main()
