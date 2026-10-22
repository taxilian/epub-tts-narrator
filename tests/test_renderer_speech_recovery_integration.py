"""Exercise automatic sentence recovery through the real render/cache workflow."""

from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import tempfile
from threading import Lock
import unittest
import wave

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, SpeechVerificationFailed, _render_chunk,
)
from test_renderer_workflow import FFMPEG, FFPROBE, timed_transcript, write_tone


TEXT = "Insang Mortuary Services! What does it—"
FIRST = "Insang Mortuary Services!"
LAST = "What does it—"


def pcm(path):
    with wave.open(str(path), "rb") as audio:
        return audio.readframes(audio.getnframes())


class RecoveryEngine:
    identity = {"engine": "recovery-fixture", "model": "offline-v1"}

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.dialogue_calls = []
        self.interrupt_part = None
        self.fail_part = None
        self.silent = False

    def _write(self, text, path, seed):
        if self.interrupt_part == text:
            raise KeyboardInterrupt("fixture interrupted part")
        if self.silent:
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                audio.writeframes(struct.pack("<" + "h" * 44100, *([0] * 44100)))
        else:
            write_tone(path, frequency=220 + len(self.calls) * 31 + len(self.dialogue_calls) * 17)
        recognized = text
        if text == TEXT or text == self.fail_part:
            recognized = text.replace("does", "is")
        self.registry[hashlib.sha256(pcm(path)).hexdigest()] = recognized
        return {"fixture": True, "seed": seed}

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.calls.append({"text": text, "voice": deepcopy(voice),
                           "seed": seed, "direction": direction})
        return self._write(text, path, seed)

    def synthesize_dialogue(self, turns, path, *, seed):
        self.dialogue_calls.append(deepcopy(turns))
        return self._write(" ".join(turn["text"].strip() for turn in turns), path, seed)


class RecoveryVerifier:
    """Read registered one-second PCM fixtures, including their exact concatenation."""

    identity = {"backend": "recovery-asr-fixture", "version": 1}

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.error = None
        self.fail_assembly = False

    def transcribe(self, path):
        self.calls.append(Path(path))
        if self.error:
            raise self.error
        data = pcm(path)
        # The fixture ASR only gets the waveform. Each normalized part is one
        # second; decoding every frame block establishes actual concatenation
        # order without passing the requested text into the recognizer.
        self_frames = 44100 * 2
        if not data or len(data) % self_frames:
            raise ValueError("unknown fixture PCM length")
        recognized = [self.registry[hashlib.sha256(data[start:start + self_frames]).hexdigest()]
                      for start in range(0, len(data), self_frames)]
        text = " ".join(recognized)
        if self.fail_assembly and len(recognized) > 1:
            text += " Thank you."
        result = timed_transcript(text)
        duration = len(recognized)
        result["segments"][0]["end"] *= duration
        for word in result["segments"][0]["words"]:
            word["start"] *= duration
            word["end"] *= duration
        return result


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for recovery integration")
class SpeechRecoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["quality"] = validate_policy({})
        self.config["render"].update(max_attempts=3, sample_rate=44100,
                                      ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.registry = {}
        self.engine = RecoveryEngine(self.registry)
        self.verifier = RecoveryVerifier(self.registry)
        self.voices = {"voice-1": {"asset_hash": "reference-asset-1", "kind": "reference",
                                    "reference_text": "Fixture voice reference.",
                                    "reference_sha256": "1" * 64}}
        self.chunk = {
            "chunk_id": "chapter-6-segment-42", "spoken_text": TEXT,
            "silent_control": False,
            "source_ranges": [{"block_id": "b000123", "start": 7, "end": 7 + len(TEXT)}],
            "segment": {"id": "segment-42", "role_id": "insang", "voice_id": "voice-1",
                        "delivery": {"direction": "startled"}},
        }
        self.messages = []

    def render_chunk(self, chunk=None, *, retry_budget=None):
        return _render_chunk(
            chunk or self.chunk, 1, 1, self.voices, self.engine, self.engine.identity,
            self.verifier, self.config, self.work, {}, Lock(),
            notify=self.messages.append,
            retry_budget=retry_budget if retry_budget is not None else SpeechRetryBudget())

    def markers(self):
        return list((self.work / "clips").glob("*/speech-recovery.json"))

    def test_failed_whole_takes_recover_exact_parts_and_publish_normal_reusable_cache(self):
        result, acceptance = self.render_chunk()
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 3 + [FIRST, LAST])
        self.assertEqual(result, {**self.chunk, "wav_path": result["wav_path"]})
        self.assertEqual(acceptance["chunk_id"], self.chunk["chunk_id"])
        self.assertEqual(acceptance["audio_sha256"], file_hash(Path(result["wav_path"])))
        report = acceptance["report"]
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["expected_text"], TEXT)
        self.assertEqual(report["transcript"], TEXT)
        self.assertEqual(report["word_timing_status"], "available")
        self.assertGreater(report["words"][-1]["end"], 1)
        self.assertEqual(report["metrics"]["duration_seconds"], 2)
        self.assertEqual(report["metrics"]["wer"], 0)
        self.assertEqual(len(self.markers()), 1)
        self.assertTrue(list((self.work / "speech-cache").glob("*.json")))
        for call in self.engine.calls:
            self.assertEqual(call["voice"], self.voices["voice-1"])
            self.assertEqual(call["direction"], "startled")
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual(record["request"]["kind"], "speech")
        self.assertEqual(record["request"]["chunk"], self.chunk)
        self.assertEqual("".join(part["text"] for part in record["request"]["recovery"]["parts"]), TEXT)
        self.assertEqual(len(record["engine_metadata"]["recovery"]["parts"]), 2)
        self.assertFalse(list(Path(result["wav_path"]).parent.rglob("*-raw.wav")),
                         "Accepted recovery parts and assembly should discard redundant raw audio")
        before_calls = (len(self.engine.calls), len(self.verifier.calls))
        resumed, resumed_acceptance = self.render_chunk()
        self.assertEqual(resumed, result)
        self.assertEqual(resumed_acceptance, acceptance)
        self.assertEqual((len(self.engine.calls), len(self.verifier.calls)), before_calls)

    def test_interrupted_second_part_resumes_first_verified_part_without_whole_takes(self):
        self.engine.interrupt_part = LAST
        with self.assertRaises(KeyboardInterrupt):
            self.render_chunk()
        self.assertEqual(len(self.markers()), 1)
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 3 + [FIRST, LAST])
        saved_parts = list((self.work / "clips").glob("*/parts/*/accepted.json"))
        self.assertEqual(len(saved_parts), 1)
        saved = {path: path.read_bytes() for path in saved_parts}
        self.engine.interrupt_part = None
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance["report"]["passed"])
        self.assertTrue(Path(result["wav_path"]).is_file())
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 3 + [FIRST, LAST, LAST])
        for path, contents in saved.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_explicit_rerun_retries_only_failed_parts_with_fresh_bounded_budget(self):
        self.engine.fail_part = LAST
        budget = SpeechRetryBudget()
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(retry_budget=budget)
        calls = list(self.engine.calls)
        self.assertEqual([call["text"] for call in calls], [TEXT] * 3 + [FIRST] + [LAST] * 3)
        saved_parts = list((self.work / "clips").glob("*/parts/*/accepted.json"))
        self.assertEqual(len(saved_parts), 1)
        saved = {path: path.read_bytes() for path in saved_parts}
        self.assertEqual(len(self.markers()), 1)
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(retry_budget=budget)
        self.assertEqual(self.engine.calls, calls)
        # A new explicit run grants a new bounded allowance to the failing
        # part, while the original whole-text strategy remains exhausted.
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual([call["text"] for call in self.engine.calls],
                         [TEXT] * 3 + [FIRST] + [LAST] * 6)
        for path, contents in saved.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))
        self.engine.fail_part = None
        result, acceptance = self.render_chunk()
        self.assertTrue(acceptance["report"]["passed"])
        self.assertTrue(Path(result["wav_path"]).is_file())
        self.assertEqual([call["text"] for call in self.engine.calls],
                         [TEXT] * 3 + [FIRST] + [LAST] * 7)
        for path, contents in saved.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_complete_join_must_also_pass_verification_before_cache_acceptance(self):
        self.verifier.fail_assembly = True
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 3 + [FIRST, LAST])
        self.assertEqual(len(list((self.work / "clips").glob("*/parts/*/accepted.json"))), 2)
        self.assertFalse(list((self.work / "clips").glob("*/accepted.json")))
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))

    def test_silent_audio_never_triggers_text_recovery(self):
        self.engine.silent = True
        with self.assertRaises(SpeechVerificationFailed) as failure:
            self.render_chunk()
        self.assertIn("silent_audio", failure.exception.report["reasons"])
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 3)
        self.assertEqual(self.verifier.calls, [])
        self.assertEqual(self.markers(), [])

    def test_asr_failure_never_triggers_text_recovery(self):
        self.verifier.error = RuntimeError("fixture recognizer unavailable")
        with self.assertRaises(SpeechVerificationFailed) as failure:
            self.render_chunk()
        self.assertTrue(any(reason.startswith("verification_failed:")
                            for reason in failure.exception.report["reasons"]))
        self.assertEqual([call["text"] for call in self.engine.calls], [TEXT] * 3)
        self.assertEqual(self.markers(), [])

    def test_paragraph_failure_stays_on_existing_paragraph_recovery_path(self):
        paragraph = deepcopy(self.chunk)
        paragraph["paragraph_turns"] = [
            {**deepcopy(self.chunk), "spoken_text": FIRST},
            {**deepcopy(self.chunk), "spoken_text": LAST},
        ]
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(paragraph)
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.markers(), [])


if __name__ == "__main__":
    unittest.main()
